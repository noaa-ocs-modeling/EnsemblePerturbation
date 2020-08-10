import os
import unittest

from EnsemblePerturbation.parse_output import ADCIRC_OUTPUT_DATA_VARIABLES, parse_adcirc_output


class TestParser(unittest.TestCase):
    def test_parse_adcirc_output(self):
        adcirc_output_directory = os.path.expanduser(r"~\Downloads\data\NetCDF_Shinnecock_Inlet")
        output_data = parse_adcirc_output(adcirc_output_directory)

        assert all(data_variable in output_data for data_variable in ADCIRC_OUTPUT_DATA_VARIABLES)


if __name__ == '__main__':
    unittest.main()
