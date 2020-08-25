from abc import ABC, abstractmethod

from geopandas import GeoDataFrame

from ensemble_perturbation import get_logger

LOGGER = get_logger('seabed')


class SeabedDescriptions(ABC):
    def __init__(self, bounds: (float, float, float, float) = None,
                 surveys: [str] = None):
        self.bounds = bounds
        self.__surveys = surveys

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
    def seabed_descriptions(self) -> [str]:
        raise NotImplementedError

    def __iter__(self) -> GeoDataFrame:
        for survey in self.surveys:
            yield self[survey]
