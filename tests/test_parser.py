from pathlib import Path
import unittest

from ensemble_perturbation.parsing.adcirc import (
    ADCIRC_OUTPUT_DATA_VARIABLES,
    parse_adcirc_output,
)

ADCIRC_OUTPUT_DIRECTORY = Path(
    __file__).parent / 'data/Shinnecock_Inlet_NetCDF_output'


class TestParser(unittest.TestCase):
    def test_parse_adcirc_output(self):
        output_data = parse_adcirc_output(ADCIRC_OUTPUT_DIRECTORY)

        for data_variable in ADCIRC_OUTPUT_DATA_VARIABLES:
            self.assertIn(data_variable, output_data)


if __name__ == '__main__':
    unittest.main()
