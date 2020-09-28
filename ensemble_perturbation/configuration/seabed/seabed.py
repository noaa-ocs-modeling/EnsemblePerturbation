from abc import ABC, abstractmethod
import os
from os import PathLike

from geopandas import GeoDataFrame
from pyproj import CRS

from ensemble_perturbation.utilities import get_logger

LOGGER = get_logger('configuration.seabed')


class SeabedDescriptions(ABC):
    longitude_field = 'Longitude'
    latitude_field = 'Latitude'
    description_field = 'Description'

    def __init__(self, bounds: (float, float, float, float) = None,
                 surveys: [str] = None, crs: CRS = None):
        self.bounds = bounds
        self.__surveys = surveys
        self.crs = CRS.from_user_input(crs) if crs is not None else None

    @classmethod
    @abstractmethod
    def all_surveys(cls) -> [str]:
        raise NotImplementedError

    @property
    def surveys(self) -> [str]:
        if self.__surveys is None:
            self.__surveys = self.__class__.all_surveys()
        return self.__surveys

    def __getitem__(self, survey: str) -> GeoDataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self) -> GeoDataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def descriptions(self) -> [str]:
        raise NotImplementedError

    def __iter__(self) -> GeoDataFrame:
        for survey in self.surveys:
            yield self[survey]

    def write(self, filename: PathLike, **kwargs):
        drivers = {
            '.csv': 'CSV',
            '.gpkg': 'GPKG',
            '.json': 'GeoJSON',
            '.shp': 'Esri Shapefile',
            '.gdb': 'OpenFileGDB',
            '.gml': 'GML',
            '.xml': 'GML'
        }

        extension = os.path.splitext(filename)[-1]
        kwargs['driver'] = drivers[extension]

        self.data.to_file(str(filename), **kwargs)
