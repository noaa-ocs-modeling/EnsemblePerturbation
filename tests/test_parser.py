import re

from ensembleperturbation.parsing.adcirc import parse_adcirc_output
from tests import DATA_DIRECTORY


def test_parse_adcirc_output():
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_parse_adcirc_output'
    output_filenames = [
        filename.name
        for filename in reference_directory.iterdir()
        if re.match('\.6(0-9)?\.nc', str(filename))
    ]

    output_data = parse_adcirc_output(reference_directory)
    for data_variable in output_filenames:
        assert data_variable in output_data
