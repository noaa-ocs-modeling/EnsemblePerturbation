import unittest

from ensemble_perturbation import repository_root
from ensemble_perturbation.outputs.comparison import ObservationComparison, \
    StationComparison

ROOT_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = ROOT_DIRECTORY / 'input'
OUTPUT_DIRECTORY = ROOT_DIRECTORY / 'output'


class TestComparison(unittest.TestCase):
    def test_base_comparison(self):
        comparison = ObservationComparison(INPUT_DIRECTORY, OUTPUT_DIRECTORY,
                                           ['u', 'v', 'zeta'])

        assert isinstance(comparison, StationComparison)


if __name__ == '__main__':
    unittest.main()
