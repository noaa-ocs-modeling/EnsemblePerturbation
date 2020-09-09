from pathlib import Path

from matplotlib import pyplot

from ensemble_perturbation import get_logger
from ensemble_perturbation.outputs.comparison import StationComparison

LOGGER = get_logger('reference.zeta')

if __name__ == '__main__':
    root_directory = Path(__file__).parent.parent

    input_directory = root_directory / 'data/input'
    output_directory = root_directory / 'data/output'

    comparison = StationComparison(input_directory, output_directory,
                                   ['zeta'])

    comparison.plot_values()
    comparison.plot_errors()
    comparison.plot_rmse()

    pyplot.show()

    print('done')
