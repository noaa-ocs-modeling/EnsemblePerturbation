import os
import unittest

from ensemble_perturbation.outputs.parse_output import \
    ADCIRC_OUTPUTS, parse_adcirc_output

ADCIRC_OUTPUT_DIRECTORY = os.path.join(__file__, os.pardir, 'data/Shinnecock_Inlet_NetCDF_output')


class TestParser(unittest.TestCase):
    def test_parse_adcirc_output(self):
        output_data = parse_adcirc_output(ADCIRC_OUTPUT_DIRECTORY)

        assert all(
            data_variable in output_data for data_variable in ADCIRC_OUTPUTS)


if __name__ == '__main__':
    unittest.main()
