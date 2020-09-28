from concurrent import futures
from functools import lru_cache

from bs4 import BeautifulSoup
import geopandas
from geopandas import GeoDataFrame
import numpy
import pandas
from pyproj import CRS, Transformer
import requests
from requests import Response, Session
from requests_futures.sessions import FuturesSession

from ensemble_perturbation.configuration.seabed.seabed import \
    SeabedDescriptions
from ensemble_perturbation.utilities import get_logger

LOGGER = get_logger('configuration.seabed.ngdc')

URL = 'https://www.ngdc.noaa.gov/geosamples/surveydisplay.jsp'
DATA_CRS = CRS.from_epsg(4326)
NO_SURVEY_OPTION = 'select a survey'


class NGDCSeabedDescriptions(SeabedDescriptions):
    def __init__(self, surveys: [str] = None,
                 bounds: (float, float, float, float) = None, crs: CRS = None):
        super().__init__(bounds, surveys,
                         crs if crs is not None else DATA_CRS)

    @classmethod
    def all_surveys(cls) -> [str]:
        page = BeautifulSoup(requests.get(URL).content, 'html.parser')
        survey_form = page.find('form', {'name': 'geographic'})
        survey_selection = survey_form.find('select', {'name': 's'})
        survey_options = survey_selection.find_all('option')
        surveys = sorted([option.text.upper() for option in survey_options
                          if option.text != NO_SURVEY_OPTION])
        subsurvey_indices = []
        for survey_index, survey in enumerate(surveys):
            for character_index in range(len(survey)):
                if survey[:character_index] in surveys:
                    subsurvey_indices.append(survey_index)
                    break
        for subsurvey_index in reversed(subsurvey_indices):
            del surveys[subsurvey_index]
        return surveys

    @property
    @lru_cache(maxsize=1)
    def descriptions(self) -> [str]:
        return sorted(self.data[self.description_field].unique())

    @lru_cache(maxsize=None)
    def __getitem__(self, survey: str) -> GeoDataFrame:
        response = self.__survey_html(survey)
        return self.__table(response.content)

    @lru_cache(maxsize=None)
    def __survey_html(self, survey: str = None,
                      session: Session = None) -> Response:
        if session is None:
            session = Session()

        query = {'s': survey}
        if self.bounds is not None:
            query.update({
                'llon': self.bounds[0], 'llat': self.bounds[1],
                'rlon': self.bounds[2], 'ulat': self.bounds[3]
            })

        return session.post(URL, data=query)

    def __table(self, html: str) -> GeoDataFrame:
        results = BeautifulSoup(html, "html.parser")
        tables = results.find_all('table', {'summary': 'layout table'})

        summary = tables[2].find_all('h1')[1].text.strip()
        try:
            num_rows = int(summary[0])
        except ValueError:
            raise ValueError('could not parse records from HTML')

        LOGGER.debug(f'parsing {num_rows} seabed descriptions from survey')

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

            longitude = numpy.array(data[self.longitude_field], dtype=float)
            latitude = numpy.array(data[self.latitude_field], dtype=float)

            data[self.longitude_field] = longitude
            data[self.latitude_field] = latitude

            if self.crs != DATA_CRS:
                transformer = Transformer.from_crs(DATA_CRS, self.crs)
                x, y = transformer.transform(longitude, latitude)
            else:
                x, y = longitude, latitude

            data['geometry'] = geopandas.points_from_xy(x, y, crs=self.crs)
            crs = self.crs
        else:
            data = {}
            crs = None

        table = GeoDataFrame(data, crs=crs)
        if self.bounds is not None and table.shape[0] > 0:
            table = table.cx[self.bounds[0]:self.bounds[2],
                    self.bounds[1]:self.bounds[3]]
        return table

    @property
    @lru_cache(maxsize=1)
    def data(self) -> GeoDataFrame:
        session = FuturesSession()
        future_responses = [self.__survey_html(survey, session)
                            for survey in self.surveys]
        tables = [self.__table(response.result().content)
                  for response in futures.as_completed(future_responses)]
        return pandas.concat(tables)
