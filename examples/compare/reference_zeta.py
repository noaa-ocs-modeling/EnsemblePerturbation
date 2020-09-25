from pathlib import Path

from matplotlib import pyplot

from ensemble_perturbation.outputs.comparison import ZetaComparison
from ensemble_perturbation.utilities import get_logger

LOGGER = get_logger('compare.zeta')

if __name__ == '__main__':
    root_directory = Path(__file__).parent.parent

    input_directory = root_directory / 'data/input'
    output_directory = root_directory / 'data/output'

    comparison = ZetaComparison(input_directory, output_directory)

    comparison.plot_values()
    comparison.plot_errors()
    comparison.plot_rmse()

    pyplot.show()

    print('done')
