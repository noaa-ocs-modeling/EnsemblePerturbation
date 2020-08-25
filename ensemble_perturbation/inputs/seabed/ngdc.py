from concurrent import futures
from functools import lru_cache

from bs4 import BeautifulSoup
from geopandas import GeoDataFrame
import numpy
import pandas
from pyproj import CRS
import requests
from requests import Response, Session
from requests_futures.sessions import FuturesSession
from shapely.geometry import Point

from ensemble_perturbation import get_logger
from ensemble_perturbation.inputs.seabed.seabed import SeabedDescriptions

LOGGER = get_logger('seabed.ngdc')


class NGDCSeabedDescriptions(SeabedDescriptions):
    url = 'https://www.ngdc.noaa.gov/geosamples/surveydisplay.jsp'
    longitude_field = 'Longitude'
    latitude_field = 'Latitude'
    description_field = 'Description'

    def __init__(self, surveys: [str] = None,
                 bounds: (float, float, float, float) = None):
        super().__init__(bounds, surveys)

    @classmethod
    def all_surveys(cls) -> [str]:
        page = BeautifulSoup(requests.get(cls.url).content, 'html.parser')
        survey_form = page.find('form', {'name': 'geographic'})
        survey_selection = survey_form.find('select', {'name': 's'})
        survey_options = survey_selection.find_all('option')
        return [option.text for option in survey_options
                if option.text != 'select a survey']

    @property
    @lru_cache(maxsize=1)
    def seabed_descriptions(self) -> [str]:
        return sorted(self.data[self.description_field].unique())

    @lru_cache(maxsize=None)
    def __getitem__(self, survey: str) -> GeoDataFrame:
        response = self.__survey_html(survey)
        return self.__table(response.content)

    @lru_cache(maxsize=None)
    def __survey_html(self, survey: str, session: Session = None) -> Response:
        if session is None:
            session = Session()
        survey = str(survey)

        bounds = {
            'llon': self.bounds[0], 'llat': self.bounds[1],
            'rlon': self.bounds[2], 'ulat': self.bounds[3]
        } if self.bounds is not None else {}

        return session.post(self.url, {'s': survey, **bounds})

    def __table(self, html: str, survey: str) -> GeoDataFrame:
        results = BeautifulSoup(html, "html.parser")
        tables = results.find_all('table', {'summary': 'layout table'})

        num_rows = int(tables[2].find_all('h1')[1].text.strip()[0])

        LOGGER.debug(f'parsing {num_rows} seabed descriptions '
                     f'from survey "{survey}"')

        if num_rows > 0:
            table = tables[-3]
            rows = table.find_all('tr')
            rows = [row.find_all('td') for row in rows]
            fields = [field.text for field in rows[0]]
            rows = rows[1:]

            data = {field_name: [] for field_name in fields}
            for row in rows:
                if len(row) == len(fields):
                    for field_index, field in enumerate(fields):
                        data[field].append(row[field_index].text)

            data[self.longitude_field] = numpy.array(
                data[self.longitude_field], dtype=float)
            data[self.latitude_field] = numpy.array(
                data[self.latitude_field], dtype=float)

            data['geometry'] = [Point(data[self.longitude_field][index],
                                      data[self.latitude_field][index])
                                for index in
                                range(len(data[self.longitude_field]))]
            data['survey'] = survey
            crs = CRS.from_epsg(4326)
        else:
            data = {}
            crs = None

        return GeoDataFrame(data, crs=crs)

    @property
    @lru_cache(maxsize=1)
    def data(self) -> GeoDataFrame:
        session = FuturesSession()
        future_responses = {self.__survey_html(survey, session): survey
                            for survey in self.surveys}
        tables = [self.__table(response.result().content,
                               future_responses[response])
                  for response in futures.as_completed(future_responses)]
        return pandas.concat(tables)
