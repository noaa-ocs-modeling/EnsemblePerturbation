from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import geopandas
from matplotlib import pyplot
from matplotlib.cm import get_cmap
import numpy
import xarray
from xarray import DataArray

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_quadrature')

if __name__ == '__main__':
    plot_storm = True
    plot_results = True
    plot_percentile = True

    save_plot = False
    show_plot = True

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
    subsetted_times = elevations['time'][::10]
    subsetted_nodes = elevations['node'][::1000]
    # samples = elevations['zeta'].sel({'time': subsetted_times, 'node': subsetted_nodes})
    # samples = elevations['zeta']
    samples = max_elevations['zeta_max'].sel({'node': subsetted_nodes})
    # samples = max_elevations['zeta_max']
    LOGGER.info(f'sample size: {samples.shape}')

    if plot_storm:
        storm_figure = pyplot.figure()
        title = f'standard deviation of {len(samples["node"])} max elevation(s) across {len(samples["run"])} run(s)'
        if 'time' in samples:
            title = f'{title} and {len(samples["time"])} time(s)'
        storm_figure.suptitle(title)
        axis = storm_figure.add_subplot(1, 1, 1)

        countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        countries.plot(color='lightgrey', ax=axis)

        if 'time' in samples:
            max_samples = samples.max('time')
        else:
            max_samples = samples

        axis.scatter(samples['x'], samples['y'], c=max_samples.std('run'))

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
                nodes=perturbations['perturbations'].T,
                weights=perturbations['weights'],
                solves=samples,
            )
        except AssertionError:
            if (
                len(perturbations['perturbations']['run'])
                != len(perturbations['weights'])
                != len(samples)
            ):
                raise AssertionError(
                    f'{len(perturbations["perturbations"]["run"])} != {len(perturbations["weights"])} != {len(samples)}'
                )
            else:
                raise

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

        predicted_mean = DataArray(
            predicted_mean, coords=reference_mean.coords, dims=reference_mean.dims
        )
        predicted_std = DataArray(
            predicted_std, coords=reference_std.coords, dims=reference_std.dims
        )

        mean_figure = pyplot.figure()
        title = (
            f'surrogate-predicted and modeled means for {len(predicted_mean["node"])} nodes'
        )
        if 'time' in predicted_mean:
            title = f'{title} over {len(predicted_mean["time"])} times'
        mean_figure.suptitle(title)

        std_figure = pyplot.figure()
        title = f'surrogate-predicted and modeled standard deviations for {len(predicted_mean["node"])} nodes'
        if 'time' in predicted_mean:
            title = f'{title} over {len(predicted_mean["time"])} times'
        std_figure.suptitle(title)

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

        if 'time' in samples:
            for node_index, node in enumerate(samples['node'].values):
                color = colors[node_index]

                predicted_node_mean = predicted_mean.sel(node=node)
                predicted_node_std = predicted_std.sel(node=node)
                reference_node_mean = reference_mean.sel(node=node)
                reference_node_std = reference_std.sel(node=node)

                mean_value_axis.plot(
                    samples['time'].values, predicted_node_mean, '--', c=color
                )
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
        else:
            mean_value_axis.bar(predicted_mean['node'], predicted_mean)
            mean_difference_axis.bar(
                predicted_mean['node'], numpy.abs(predicted_mean - reference_mean)
            )
            std_value_axis.bar(predicted_std['node'], predicted_std)
            std_difference_axis.bar(
                predicted_std['node'], numpy.abs(predicted_std - reference_std)
            )

        if save_plot:
            mean_figure.savefig(input_directory / 'mean.png', bbox_inches='tight')
            std_figure.savefig(input_directory / 'std.png', bbox_inches='tight')

    percentiles = [90]
    percentile_filename = input_directory / 'percentiles.nc'
    if not percentile_filename.exists():
        LOGGER.info(f'calculating {len(percentiles)} percentiles: {percentiles}')
        predicted_percentiles = chaospy.Perc(
            poly=surrogate_model, q=percentiles, dist=distribution, sample=samples.shape[1],
        )
        LOGGER.info(f'saving percentiles to "{percentile_filename}"')

        predicted_percentiles = DataArray(
            predicted_percentiles,
            coords={'percentile': percentiles, 'node': samples['node']},
            dims=('percentile', 'node'),
            name='percentiles',
        )

        predicted_percentiles.to_netcdf(percentile_filename)
    else:
        LOGGER.info(f'loading percentiles from "{percentile_filename}"')
        predicted_percentiles = xarray.open_dataset(percentile_filename)

    if plot_percentile:
        percentile_figure = pyplot.figure()

        for percentile_index, percentile_output in enumerate(
            predicted_percentiles['percentiles']
        ):
            axis = percentile_figure.add_subplot(len(percentiles), 1, percentile_index + 1)

            if 'time' in percentile_output:
                for node_index in range(percentile_output.shape[-1]):
                    node_percentile = percentile_output[:, node_index]
                    axis.plot(samples['time'].values, node_percentile)
            else:
                axis.bar(samples['node'], percentile_output)

            title = f'{percentile_output["percentile"].values} percentile for {len(percentile_output["node"])} nodes'
            if 'time' in percentile_output:
                title = f'{title} over {len(percentile_output["time"])} times'
            percentile_figure.suptitle(title)

        if save_plot:
            percentile_figure.savefig(input_directory / 'percentiles.png', bbox_inches='tight')

    if show_plot:
        pyplot.show()
