#! /usr/bin/env python

from pathlib import Path

from matplotlib import pyplot

from ensembleperturbation.parsing.comparison import VelocityComparison
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('compare.uv')

if __name__ == '__main__':
    root_directory = Path(__file__).parent.parent

    input_directory = root_directory / 'data/input'
    output_directory = root_directory / 'data/output'

    comparison = VelocityComparison(input_directory, output_directory)

    comparison.plot_values()
    comparison.plot_errors()
    comparison.plot_rmse()

    pyplot.show()

    print('done')
