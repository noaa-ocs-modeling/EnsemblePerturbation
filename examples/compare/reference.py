#! /usr/bin/env python

from pathlib import Path

from matplotlib import pyplot

from ensembleperturbation.outputs.comparison import ObservationStationComparison
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('reference.uv')

if __name__ == '__main__':
    root_directory = Path(__file__).parent.parent

    input_directory = root_directory / 'data/input'
    output_directory = root_directory / 'data/output'

    comparison = ObservationStationComparison(
        input_directory, output_directory, ['u', 'v', 'zeta']
    )

    comparison.plot_values()
    comparison.plot_errors()
    comparison.plot_rmse()

    pyplot.show()

    print('done')
