#! /usr/bin/env python

from pathlib import Path
import sys

from matplotlib import pyplot

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from ensembleperturbation.parsing.comparison import ZetaComparison
from ensembleperturbation.utilities import get_logger

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
