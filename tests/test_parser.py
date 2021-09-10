import re

from ensembleperturbation.parsing.adcirc import combine_outputs, parse_adcirc_outputs
from tests import check_reference_directory, DATA_DIRECTORY


def test_parse_adcirc_output():
    input_directory = DATA_DIRECTORY / 'input' / 'test_parse_adcirc_output'
    output_filenames = [
        filename.name
        for filename in input_directory.iterdir()
        if re.match('\.6(0-9)?\.nc', str(filename))
    ]

    output_data = parse_adcirc_outputs(input_directory)
    for data_variable in output_filenames:
        assert data_variable in output_data


def test_combine_output():
    input_directory = DATA_DIRECTORY / 'input' / 'test_combine_outputs'
    output_directory = DATA_DIRECTORY / 'output' / 'test_combine_outputs'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_combine_outputs'
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    file_data_types = {
        'fort.63.nc': ['zeta'],
        'fort.64.nc': None,
        'maxele.63.nc': ['zeta_max'],
        'maxvel.63.nc': ['vel_max'],
    }

    output_filename = output_directory / 'outputs.h5'

    combine_outputs(
        input_directory,
        file_data_variables=file_data_types,
        maximum_depth=5.0,
        output_filename=output_filename,
    )

    check_reference_directory(output_directory, reference_directory)
