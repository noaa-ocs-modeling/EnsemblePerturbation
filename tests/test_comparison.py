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
    input_directory = INPUT_DIRECTORY / 'test_parse_adcirc_output'
    output_directory = input_directory

    comparison = ObservationStationComparison(
        input_directory, output_directory, ['u', 'v', 'zeta']
    )

    assert isinstance(comparison, StationComparison)


def test_virtual_stations():
    input_directory = INPUT_DIRECTORY / 'test_parse_adcirc_output'
    output_directory = input_directory

    stations_filename = ROOT_DIRECTORY / 'virtual_stations.gpkg'

    comparison = VirtualStationComparison(
        input_directory, output_directory, ['u', 'v', 'zeta'], stations_filename
    )

    assert isinstance(comparison, StationComparison)
