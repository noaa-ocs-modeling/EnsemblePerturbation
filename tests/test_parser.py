from pathlib import Path

from ensembleperturbation.parsing.adcirc import ADCIRC_OUTPUTS, parse_adcirc_output

ADCIRC_OUTPUT_DIRECTORY = Path(__file__).parent / 'data/Shinnecock_Inlet_NetCDF_output'


def test_parse_adcirc_output():
    output_data = parse_adcirc_output(ADCIRC_OUTPUT_DIRECTORY)

    assert all(data_variable in output_data for data_variable in ADCIRC_OUTPUTS)
