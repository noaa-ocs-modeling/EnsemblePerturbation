import unittest

from ensemble_perturbation import repository_root
from ensemble_perturbation.outputs.comparison import \
    ObservationStationComparison, \
    StationComparison, VirtualStationComparison

ROOT_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = ROOT_DIRECTORY / 'input'
OUTPUT_DIRECTORY = ROOT_DIRECTORY / 'output'


class TestComparison(unittest.TestCase):
    def test_observation(self):
        comparison = ObservationStationComparison(INPUT_DIRECTORY,
                                                  OUTPUT_DIRECTORY,
                                                  ['u', 'v', 'zeta'])

        assert isinstance(comparison, StationComparison)

    def test_virtual_stations(self):
        stations_filename = ROOT_DIRECTORY / 'virtual_stations.gpkg'

        comparison = VirtualStationComparison(INPUT_DIRECTORY,
                                              OUTPUT_DIRECTORY,
                                              ['u', 'v', 'zeta'],
                                              stations_filename)

        assert isinstance(comparison, StationComparison)


if __name__ == '__main__':
    unittest.main()
