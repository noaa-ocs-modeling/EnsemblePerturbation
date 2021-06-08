from ensembleperturbation.outputs.comparison import (
    ObservationStationComparison,
    StationComparison,
    VirtualStationComparison,
)
from ensembleperturbation.utilities import repository_root

ROOT_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = ROOT_DIRECTORY / 'input'
OUTPUT_DIRECTORY = ROOT_DIRECTORY / 'output'


def test_observation():
    comparison = ObservationStationComparison(
        INPUT_DIRECTORY, OUTPUT_DIRECTORY, ['u', 'v', 'zeta']
    )

    assert isinstance(comparison, StationComparison)


def test_virtual_stations():
    stations_filename = ROOT_DIRECTORY / 'virtual_stations.gpkg'

    comparison = VirtualStationComparison(
        INPUT_DIRECTORY, OUTPUT_DIRECTORY, ['u', 'v', 'zeta'], stations_filename
    )

    assert isinstance(comparison, StationComparison)
