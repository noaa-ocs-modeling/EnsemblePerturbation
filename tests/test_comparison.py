from pathlib import Path
import unittest

from ensemble_perturbation.outputs.comparison import ReferenceComparison, \
    VelocityComparison, ZetaComparison


class TestComparison(unittest.TestCase):
    def test_base_comparison(self):
        root_directory = Path(__file__).parent.parent

        input_directory = root_directory / 'data/input'
        output_directory = root_directory / 'data/output'

        comparison = ReferenceComparison(input_directory, output_directory,
                                         ['u', 'v', 'zeta'])

        assert isinstance(comparison, ReferenceComparison)

    def test_zeta_comparison(self):
        root_directory = Path(__file__).parent.parent

        input_directory = root_directory / 'data/input'
        output_directory = root_directory / 'data/output'

        comparison = ZetaComparison(input_directory, output_directory)

        assert isinstance(comparison, ReferenceComparison)

    def test_velocity_comparison(self):
        root_directory = Path(__file__).parent.parent

        input_directory = root_directory / 'data/input'
        output_directory = root_directory / 'data/output'

        comparison = VelocityComparison(input_directory, output_directory)

        assert isinstance(comparison, ReferenceComparison)


if __name__ == '__main__':
    unittest.main()
