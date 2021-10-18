from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import geopandas
from matplotlib import pyplot
from matplotlib.cm import get_cmap
import numpy
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_quadrature')

if __name__ == '__main__':
    plot_storm = True
    plot_results = True
    plot_percentile = True

    save_plot = True
    show_plot = False

    input_directory = Path.cwd()
    surrogate_filename = input_directory / 'surrogate.npy'

    filenames = ['perturbations.nc', 'fort.63.nc', 'maxele.63.nc']

    datasets = {}
    existing_filenames = []
    for filename in filenames:
        filename = input_directory / filename
        if filename.exists():
            datasets[filename.name] = xarray.open_dataset(filename)
            existing_filenames.append(filename.name)

    for filename in existing_filenames:
        filenames.remove(filename)

    if len(filenames) > 0:
        datasets.update(
            combine_outputs(
                input_directory,
                file_data_variables=filenames,
                maximum_depth=0,
                only_inundated=True,
                parallel=True,
            )
        )

    perturbations = datasets['perturbations.nc']
    elevations = datasets['fort.63.nc']
    max_elevations = datasets['maxele.63.nc']

    variables = {
        variable_class.name: variable_class()
        for variable_class in VortexPerturbedVariable.__subclasses__()
    }

    distribution = chaospy.J(
        *(
            variables[variable_name].chaospy_distribution()
            for variable_name in perturbations['variable'].values
        )
    )

    # sample times and nodes
    # TODO: sample based on sentivity / eigenvalues
    # sample_times = elevations['time'][::10]
    # sample_nodes = elevations['node'][::1000]
    # samples = elevations['zeta'].loc[{'time': sample_times, 'node': sample_nodes}]
    # samples = elevations['zeta']
    samples = max_elevations['zeta_max']
    LOGGER.info(f'sample size: {samples.shape}')

    if plot_storm:
        storm_figure = pyplot.figure()
        storm_figure.suptitle(
            f'standard deviation of {len(samples["node"])} max elevation(s) across {len(samples["run"])} run(s) and {len(samples["time"])} time(s)'
        )
        axis = storm_figure.add_subplot(1, 1, 1)

        countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        countries.plot(color='lightgrey', ax=axis)

        axis.scatter(samples['x'], samples['y'], c=samples.max('time').std('run'))

        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')
        storm.data.plot(x='longitude', y='latitude', ax=axis)

        if save_plot:
            storm_figure.savefig(input_directory / 'storm.png', bbox_inches='tight')

    if not surrogate_filename.exists():
        # expand polynomials with polynomial chaos
        polynomials = chaospy.generate_expansion(
            order=3, dist=distribution, rule='three_terms_recurrence',
        )

        # create surrogate models for selected nodes
        LOGGER.info(f'fitting surrogate to {samples.shape} samples')
        try:
            surrogate_model = chaospy.fit_quadrature(
                orth=polynomials,
                nodes=perturbations['perturbations'].T.values,
                weights=perturbations['weights'].values,
                solves=samples.sel(run=samples['run'] != 'original'),
            )
        except AssertionError:
            raise AssertionError(
                f'{perturbations["perturbations"].T.shape[1]} != {len(perturbations["weights"])} != {len(samples)}'
            )

        with open(surrogate_filename, 'wb') as surrogate_file:
            LOGGER.info(f'saving surrogate model to "{surrogate_filename}"')
            surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{surrogate_filename}"')
        surrogate_model = chaospy.load(surrogate_filename, allow_pickle=True)

    if plot_results:
        LOGGER.info(f'running surrogate')
        predicted_mean = chaospy.E(poly=surrogate_model, dist=distribution)
        predicted_std = chaospy.Std(poly=surrogate_model, dist=distribution)
        reference_mean = samples.mean('run')
        reference_std = samples.std('run')

        mean_figure = pyplot.figure()
        mean_figure.suptitle(
            f'surrogate-predicted and modeled means for {predicted_mean.shape[1]} nodes over {predicted_mean.shape[0]} times'
        )

        std_figure = pyplot.figure()
        std_figure.suptitle(
            f'surrogate-predicted and modeled standard deviations for {predicted_mean.shape[1]} nodes over {predicted_mean.shape[0]} times'
        )

        colors = [
            get_cmap('gist_rainbow')(color_index / len(samples['node']))
            for color_index in range(len(samples['node']))
        ]

        mean_value_axis = mean_figure.add_subplot(2, 1, 1)
        mean_difference_axis = mean_figure.add_subplot(2, 1, 2)
        std_value_axis = std_figure.add_subplot(2, 1, 1)
        std_difference_axis = std_figure.add_subplot(2, 1, 2)

        mean_value_axis.set_title('means')
        std_value_axis.set_title('standard deviations')
        mean_difference_axis.set_title('differences')
        std_difference_axis.set_title('differences')

        for node_index in range(len(samples['node'])):
            color = colors[node_index]

            predicted_node_mean = predicted_mean[:, node_index]
            predicted_node_std = predicted_std[:, node_index]
            reference_node_mean = reference_mean[:, node_index]
            reference_node_std = reference_std[:, node_index]

            mean_value_axis.plot(samples['time'].values, predicted_node_mean, '--', c=color)
            mean_value_axis.plot(samples['time'].values, reference_node_mean, c=color)
            mean_value_axis.fill_between(
                samples['time'].values,
                predicted_node_mean - predicted_node_std,
                predicted_node_mean + predicted_node_std,
                color=color,
                alpha=0.5,
            )
            mean_value_axis.fill_between(
                samples['time'].values,
                reference_node_mean - reference_node_std,
                reference_node_mean + reference_node_std,
                color=color,
                alpha=0.25,
            )

            mean_difference_axis.plot(
                samples['time'].values,
                numpy.abs(reference_node_mean - predicted_node_mean),
                c=color,
            )

            std_value_axis.plot(samples['time'].values, reference_node_std, c=color)
            std_value_axis.plot(samples['time'].values, predicted_node_std, '--', c=color)

            std_difference_axis.plot(
                samples['time'].values, predicted_node_std - reference_node_std, c=color
            )

        if save_plot:
            mean_figure.savefig(input_directory / 'mean.png', bbox_inches='tight')
            std_figure.savefig(input_directory / 'std.png', bbox_inches='tight')

    percentiles = [90]
    percentile_filename = input_directory / 'percentiles.npy'
    if not percentile_filename.exists():
        LOGGER.info(f'calculating {len(percentiles)} percentiles: {percentiles}')
        predicted_percentiles = chaospy.Perc(
            poly=surrogate_model, q=percentiles, dist=distribution, sample=samples.shape[1],
        )
        LOGGER.info(f'saving percentiles to "{percentile_filename}"')
        numpy.save(str(percentile_filename), predicted_percentiles)
    else:
        LOGGER.info(f'loading percentiles from "{percentile_filename}"')
        predicted_percentiles = numpy.load(str(percentile_filename), allow_pickle=True)

    if plot_percentile:
        percentile_figure = pyplot.figure()

        for percentile_index, percentile_output in enumerate(predicted_percentiles):
            axis = percentile_figure.add_subplot(len(percentiles), 1, percentile_index + 1)

            percentile = percentiles[percentile_index]

            for node_index in range(percentile_output.shape[1]):
                node_percentile = percentile_output[:, node_index]
                axis.plot(samples['time'].values, node_percentile)

            percentile_figure.suptitle(
                f'{percentile} percentile for {percentile_output.shape[1]} nodes over {percentile_output.shape[0]} times'
            )

        if save_plot:
            percentile_figure.savefig(input_directory / 'percentiles.png', bbox_inches='tight')

    if show_plot:
        pyplot.show()
