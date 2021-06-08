from pathlib import Path

from ensembleperturbation.parsing.adcirc import ADCIRC_OUTPUTS, parse_adcirc_output

ADCIRC_OUTPUT_DIRECTORY = Path(__file__).parent / 'data/Shinnecock_Inlet_NetCDF_output'


class TestParser(unittest.TestCase):
    def test_parse_adcirc_output(self):
        output_data = parse_adcirc_output(ADCIRC_OUTPUT_DIRECTORY)

        assert all(data_variable in output_data for data_variable in ADCIRC_OUTPUTS)


if __name__ == '__main__':
    unittest.main()
