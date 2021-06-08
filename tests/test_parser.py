from pathlib import Path

from ensembleperturbation.parsing.adcirc import (
    ADCIRC_OUTPUT_DATA_VARIABLES,
    parse_adcirc_output,
)

ADCIRC_OUTPUT_DIRECTORY = Path(__file__).parent / 'data/Shinnecock_Inlet_NetCDF_output'


def test_parse_adcirc_output():
    output_data = parse_adcirc_output(ADCIRC_OUTPUT_DIRECTORY)
    for data_variable in ADCIRC_OUTPUT_DATA_VARIABLES:
        assert data_variable in output_data
