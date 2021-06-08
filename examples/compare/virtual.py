from ensemble_perturbation import get_logger, repository_root
from matplotlib import pyplot

from ensembleperturbation.outputs.comparison import VirtualStationComparison

LOGGER = get_logger('reference.uv')

if __name__ == '__main__':
    root_directory = repository_root() / 'examples/data'

    input_directory = root_directory / 'input'
    output_directory = root_directory / 'output'

    stations_filename = root_directory / 'virtual_stations.gpkg'

    comparison = VirtualStationComparison(
        input_directory, output_directory, ['u', 'v', 'zeta'], stations_filename
    )

    comparison.plot_values()
    comparison.plot_errors()
    comparison.plot_rmse()

    pyplot.show()

    print('done')
