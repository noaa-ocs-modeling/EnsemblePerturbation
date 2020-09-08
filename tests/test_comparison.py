import unittest

from ensemble_perturbation import repository_root
from ensemble_perturbation.outputs.comparison import ReferenceComparison, \
    VelocityComparison, ZetaComparison

ROOT_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = ROOT_DIRECTORY / 'input'
OUTPUT_DIRECTORY = ROOT_DIRECTORY / 'output'


class TestComparison(unittest.TestCase):
    def test_base_comparison(self):
        comparison = ReferenceComparison(INPUT_DIRECTORY, OUTPUT_DIRECTORY,
                                         ['u', 'v', 'zeta'])

        assert isinstance(comparison, ReferenceComparison)

    def test_zeta_comparison(self):
        comparison = ZetaComparison(INPUT_DIRECTORY, OUTPUT_DIRECTORY)

        assert isinstance(comparison, ReferenceComparison)

    def test_velocity_comparison(self):
        comparison = VelocityComparison(INPUT_DIRECTORY, OUTPUT_DIRECTORY)

        assert isinstance(comparison, ReferenceComparison)


if __name__ == '__main__':
    unittest.main()
