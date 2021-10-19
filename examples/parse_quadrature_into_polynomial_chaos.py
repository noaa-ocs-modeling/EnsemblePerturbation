from os import PathLike
from pathlib import Path

from adcircpy.forcing import BestTrackForcing
from adcircpy.forcing.winds.best_track import VortexForcing
import chaospy
import geopandas
from matplotlib import cm, gridspec, pyplot
from shapely.geometry import LineString, Point
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.uncertainty_quantification.quadrature import (
    fit_surrogate_to_quadrature,
    get_percentiles,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_nodes')


def plot_nodes_across_runs(
    nodes: xarray.Dataset,
    title: str = None,
    node_colors: [(float, float, float)] = None,
    storm: str = None,
    output_filename: PathLike = None,
):
    figure = pyplot.figure()
    if title is not None:
        figure.suptitle(title)

    grid = gridspec.GridSpec(len(nodes.data_vars), 2, figure=figure)

    map_bounds = [
        float(samples.coords['x'].min().values),
        float(samples.coords['y'].min().values),
        float(samples.coords['x'].max().values),
        float(samples.coords['y'].max().values),
    ]

    countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    if storm is None:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')
    elif not isinstance(storm, VortexForcing):
        storm = BestTrackForcing(storm)

    edge_colors = [
        cm.get_cmap('gist_rainbow')(color_index / len(nodes['node']))
        for color_index in range(len(nodes['node']))
    ]

    if node_colors is None:
        node_colors = edge_colors
    elif isinstance(node_colors, str):
        node_colors = nodes[node_colors]
        if len(node_colors.dims) > 1:
            node_colors = node_colors.mean([dim for dim in node_colors.dims if dim != 'node'])
        node_colors = node_colors.values

    map_axis = figure.add_subplot(grid[:, 0])
    countries.plot(color='lightgrey', ax=map_axis)
    storm.data.plot(
        x='longitude',
        y='latitude',
        ax=map_axis,
        label=storm_name,
        legend=storm_name is not None,
    )

    nodes.plot.scatter(x='x', y='y', c=node_colors, edgecolors=edge_colors)

    map_axis.set_xlim(
        map_bounds[0] - abs(map_bounds[0] * 0.1), map_bounds[2] + abs(map_bounds[2] * 0.1)
    )
    map_axis.set_ylim(
        map_bounds[1] - abs(map_bounds[1] * 0.1), map_bounds[3] + abs(map_bounds[3] * 0.1)
    )

    for variable_index, (variable_name, variable) in enumerate(nodes.data_vars.items()):
        variable_axis = figure.add_subplot(grid[variable_index, 1])

        if 'source' in nodes.dims:
            sources = ['model', 'surrogate']
        else:
            sources = [None]

        for source in sources:
            kwargs = {}

            if source == 'surrogate':
                kwargs['alpha'] = 0.3
                kwargs['linestyle'] = '--'

            if 'source' in nodes.dims:
                source_data = variable.sel(source=source)
            else:
                source_data = variable

            if 'time' in nodes.dims:
                for node_index in range(len(nodes['node'])):
                    node_data = source_data.isel(node=node_index)
                    node_color = node_colors[node_index]
                    node_data.plot.line(
                        x='time', c=node_color, ax=variable_axis, **kwargs,
                    )
                    if variable_name == 'mean' and 'std' in nodes.data_vars:
                        std_data = nodes['std'].sel(source=source).isel(node=node_index)
                        variable_axis.fill_between(
                            samples['time'],
                            node_data - std_data,
                            node_data + std_data,
                            color=node_color,
                            **{
                                key: value
                                if key != 'alpha'
                                else (0.3 if source == 'surrogate' else 0.6)
                                for key, value in kwargs.items()
                            },
                        )
            else:
                if variable_name == 'mean' and 'std' in nodes.data_vars:
                    kwargs['yerr'] = nodes['std'].sel(source=source)
                if source == 'surrogate':
                    kwargs['edgecolor'] = 'k'

                source_data.to_series().plot.bar(
                    x='node', color=node_colors, ax=variable_axis, **kwargs
                )

        variable_axis.set_title(variable_name)
        variable_axis.tick_params(axis='x', which='both', labelsize=6)
        variable_axis.set(xlabel=None)

    if output_filename is not None:
        figure.set_size_inches(12, 12 / 1.61803398875)
        figure.savefig(output_filename, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot_results = True
    plot_percentile = True

    save_plots = True
    show_plots = True

    storm_name = None

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
    samples = elevations['zeta'].sel({'time': subsetted_times, 'node': subsetted_nodes})
    # samples = elevations['zeta']
    # samples = max_elevations['zeta_max'].sel({'node': subsetted_nodes})
    # samples = max_elevations['zeta_max']
    LOGGER.info(f'sample size: {samples.shape}')

    if storm_name is not None:
        storm = BestTrackForcing(storm_name)
    else:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')

    # sort samples by ascending distance to storm track
    storm_linestring = LineString(list(zip(storm.data['longitude'], storm.data['latitude'])))
    distances = {
        int(node.values): storm_linestring.distance(Point(node['x'], node['y']))
        for node in samples['node']
    }
    samples = samples.assign_coords({'distance_to_track': ('node', list(distances.values()))})
    samples = samples.sortby('distance_to_track')

    if not surrogate_filename.exists():
        # expand polynomials with polynomial chaos
        polynomials = chaospy.generate_expansion(
            order=3, dist=distribution, rule='three_terms_recurrence',
        )

        surrogate_model = fit_surrogate_to_quadrature(
            samples=samples,
            polynomials=polynomials,
            perturbations=perturbations['perturbations'],
            weights=perturbations['weights'],
        )

        with open(surrogate_filename, 'wb') as surrogate_file:
            LOGGER.info(f'saving surrogate model to "{surrogate_filename}"')
            surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{surrogate_filename}"')
        surrogate_model = chaospy.load(surrogate_filename, allow_pickle=True)

    percentiles = [90]
    percentile_filename = input_directory / 'percentiles.nc'
    if not percentile_filename.exists():
        predicted_percentiles = get_percentiles(
            samples=samples,
            percentiles=percentiles,
            surrogate_model=surrogate_model,
            distribution=distribution,
        )

        predicted_percentiles = predicted_percentiles.to_dataset(name='percentiles')

        LOGGER.info(f'saving percentiles to "{percentile_filename}"')
        predicted_percentiles.to_netcdf(percentile_filename)
    else:
        LOGGER.info(f'loading percentiles from "{percentile_filename}"')
        predicted_percentiles = xarray.open_dataset(percentile_filename)

    if plot_results:
        LOGGER.info(f'running surrogate on {samples.shape} samples')
        predicted_mean = chaospy.E(poly=surrogate_model, dist=distribution)
        predicted_std = chaospy.Std(poly=surrogate_model, dist=distribution)
        reference_mean = samples.mean('run')
        reference_std = samples.std('run')

        predicted_mean = xarray.DataArray(
            predicted_mean, coords=reference_mean.coords, dims=reference_mean.dims,
        )
        predicted_std = xarray.DataArray(
            predicted_std, coords=reference_std.coords, dims=reference_std.dims,
        )

        node_results = xarray.Dataset(
            {
                'mean': xarray.combine_nested(
                    [predicted_mean, reference_mean], concat_dim='source'
                ).assign_coords({'source': ['surrogate', 'model']}),
                'std': xarray.combine_nested(
                    [predicted_std, reference_std], concat_dim='source'
                ).assign_coords({'source': ['surrogate', 'model']}),
            }
        )

        plot_nodes_across_runs(
            node_results,
            title=f'surrogate-predicted and modeled elevations for {len(node_results["node"])} nodes',
            node_colors='std',
            storm=storm,
            output_filename=input_directory / 'elevations.png' if save_plots else None,
        )

    if plot_percentile:
        percentiles = xarray.Dataset(
            {
                str(float(percentile.values)): predicted_percentiles['percentiles'].sel(
                    percentile=percentile
                )
                for percentile in predicted_percentiles['percentile']
            },
            coords=predicted_percentiles.coords,
        )

        plot_nodes_across_runs(
            percentiles,
            title=f'surrogate-predicted and modeled percentiles for {len(percentiles["node"])} nodes',
            node_colors='90',
            storm=storm,
            output_filename=input_directory / 'percentiles.png' if save_plots else None,
        )

    if show_plots:
        pyplot.show()
