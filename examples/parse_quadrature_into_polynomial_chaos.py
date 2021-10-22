import math
from os import PathLike
from pathlib import Path

from adcircpy.forcing import BestTrackForcing
from adcircpy.forcing.winds.best_track import VortexForcing
import chaospy
import geopandas
from geopandas import GeoDataFrame
from matplotlib import cm, colors, gridspec, pyplot
import numpy
from shapely.geometry import LineString
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs, FieldOutput
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

    map_axis = figure.add_subplot(grid[:, 0])
    map_title = f'{len(nodes["node"])} nodes'

    if node_colors is None:
        color_map = cm.get_cmap('gist_rainbow')
        color_values = numpy.arange(len(nodes['node'])) / len(nodes['node'])
        node_colors = color_map(color_values)
        min_value = numpy.min(color_values.values)
        max_value = numpy.max(color_values.values)
        normalization = colors.Normalize(vmin=min_value, vmax=max_value)
    elif isinstance(node_colors, str):
        color_map = cm.get_cmap('cool')
        map_title = f'{map_title} colored by "{node_colors}"'
        color_values = nodes[node_colors]
        if len(color_values.dims) > 1:
            color_values = color_values.mean(
                [dim for dim in color_values.dims if dim != 'node']
            )
        min_value = numpy.min(color_values.values)
        max_value = numpy.max(color_values.values)
        normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
        colorbar = figure.colorbar(
            mappable=cm.ScalarMappable(cmap=color_map, norm=normalization,), ax=map_axis,
        )
        colorbar.set_label(node_colors)
        color_values = (color_values - min_value) / (max_value - min_value)
        node_colors = color_map(color_values)
    else:
        color_map = cm.get_cmap('cool')
        min_value = numpy.min(node_colors)
        max_value = numpy.max(node_colors)
        color_values = (node_colors - min_value) / (max_value - min_value)
        normalization = colors.Normalize(vmin=min_value, vmax=max_value)

    countries.plot(color='lightgrey', ax=map_axis)
    storm.data.plot(
        x='longitude',
        y='latitude',
        ax=map_axis,
        label=storm_name,
        legend=storm_name is not None,
    )

    nodes.plot.scatter(x='x', y='y', c=node_colors, s=2, norm=normalization)

    map_axis.set_xlim(map_bounds[0], map_bounds[2])
    map_axis.set_ylim(map_bounds[1], map_bounds[3])
    map_axis.set_title(map_title)

    shared_axis = None
    for variable_index, (variable_name, variable) in enumerate(nodes.data_vars.items()):
        axis_kwargs = {}
        if shared_axis is not None:
            axis_kwargs['sharex'] = shared_axis

        variable_axis = figure.add_subplot(grid[variable_index, 1], **axis_kwargs)

        if shared_axis is None:
            shared_axis = variable_axis

        if variable_index < len(nodes.data_vars) - 1:
            variable_axis.get_xaxis().set_visible(False)

        if 'source' in nodes.dims:
            sources = ['model', 'surrogate']
        else:
            sources = [None]

        for source_index, source in enumerate(sources):
            if source is None or source == 'model':
                color_map = cm.get_cmap('cool')
            elif source == 'surrogate':
                color_map = cm.get_cmap('hot')
            node_colors = color_map(color_values)

            kwargs = {'norm': normalization}

            if source == 'surrogate':
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
                            alpha=0.3,
                            **kwargs,
                        )
            else:
                # if variable_name == 'mean' and 'std' in nodes.data_vars:
                #     kwargs['yerr'] = nodes['std'].sel(source=source)
                # if source == 'surrogate':
                #     kwargs['edgecolor'] = 'k'

                bar_width = 0.01
                bar_offset = bar_width * (source_index + 0.5 - len(sources) / 2)

                variable_axis.bar(
                    x=source_data['distance_to_track'] + bar_offset,
                    width=bar_width,
                    height=source_data.values,
                    color=node_colors,
                    **kwargs,
                )

        variable_axis.set_title(variable_name)
        variable_axis.tick_params(axis='x', which='both', labelsize=6)
        variable_axis.set(xlabel=None)

    if output_filename is not None:
        figure.set_size_inches(12, 12 / 1.61803398875)
        figure.savefig(output_filename, dpi=300, bbox_inches='tight')


def plot_perturbed_variables(
    perturbations: xarray.Dataset, title: str = None, output_filename: PathLike = None,
):
    figure = pyplot.figure()
    if title is None:
        title = f'{len(perturbations["run"])} pertubation(s) of {len(perturbations["variable"])} variable(s)'
    figure.suptitle(title)

    variables = perturbations['variable'].values
    num_variables = len(variables)
    num_plots = int(num_variables * (num_variables - 1) / 2)

    grid_length = math.ceil(num_plots ** 0.5)
    grid = gridspec.GridSpec(grid_length, grid_length, figure=figure, wspace=0, hspace=0)

    shared_grid_columns = {'x': {}, 'y': {}}
    for plot_index in range(num_plots):
        original_index = plot_index
        if plot_index >= num_variables:
            # TODO fix this
            plot_index = (plot_index % num_variables) + 1 if original_index % 2 else 3

        variable = variables[plot_index]

        if original_index % 2 == 0:
            shared_grid_columns['x'][variable] = None
        else:
            shared_grid_columns['y'][variable] = None

    color_map_axis = figure.add_subplot(grid[-1, -1])
    color_map_axis.set_visible(False)
    color_map = cm.get_cmap('jet')
    min_value = numpy.min(perturbations['weights'].values)
    max_value = numpy.max(perturbations['weights'].values)
    normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
    colorbar = pyplot.colorbar(
        mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
        orientation='horizontal',
        ax=color_map_axis,
    )
    colorbar.set_label('weight')

    for column_index in range(grid_length):
        column_variable = list(shared_grid_columns['y'])[column_index]
        for row_index in range(grid_length - column_index):
            row_variable = list(shared_grid_columns['x'])[row_index]

            sharex = shared_grid_columns['x'][row_variable]
            sharey = shared_grid_columns['y'][column_variable]

            variable_axis = figure.add_subplot(
                grid[row_index, column_index], sharex=sharex, sharey=sharey,
            )

            if sharex is None:
                shared_grid_columns['x'][row_variable] = variable_axis
            if sharey is None:
                shared_grid_columns['y'][column_variable] = variable_axis

            variable_axis.scatter(
                perturbations['perturbations'].sel(variable=column_variable),
                perturbations['perturbations'].sel(variable=row_variable),
                c=perturbations['weights'],
                cmap=color_map,
                norm=normalization,
            )

            if row_index == 0:
                variable_axis.set_xlabel(column_variable)
                variable_axis.xaxis.set_label_position('top')
                variable_axis.xaxis.tick_top()
                if row_index == grid_length - column_index - 1:
                    variable_axis.secondary_xaxis('bottom')
            elif row_index != grid_length - column_index - 1:
                variable_axis.xaxis.set_visible(False)

            if column_index == 0:
                variable_axis.set_ylabel(row_variable)
                if row_index == grid_length - column_index - 1:
                    variable_axis.secondary_yaxis('right')
            elif row_index == grid_length - column_index - 1:
                variable_axis.yaxis.tick_right()
            else:
                variable_axis.yaxis.set_visible(False)

    if output_filename is not None:
        figure.set_size_inches(12, 12 / 1.61803398875)
        figure.savefig(output_filename, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot_perturbations = True
    plot_results = True
    plot_percentile = True

    save_plots = True
    show_plots = False

    storm_name = None

    input_directory = Path.cwd()
    surrogate_filename = input_directory / 'surrogate.npy'

    filenames = ['perturbations.nc', 'fort.63.nc', 'maxele.63.nc']

    datasets = {}
    existing_filenames = []
    for filename in filenames:
        filename = input_directory / filename
        if filename.exists():
            datasets[filename.name] = xarray.open_dataset(filename, chunks=-1)
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

    if plot_perturbations:
        plot_perturbed_variables(
            perturbations, output_filename=input_directory / 'perturbations.png'
        )

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
    subset_bounds = (-83, 25, -72, 42)
    subsetted_nodes = elevations['node'].sel(
        node=FieldOutput.subset(elevations['node'], bounds=subset_bounds,)
    )
    # subsetted_times = elevations['time'][::10]
    # samples = elevations['zeta'].sel({'time': subsetted_times, 'node': subsetted_nodes})
    # samples = elevations['zeta']
    samples = max_elevations['zeta_max'].sel(node=subsetted_nodes)
    # samples = max_elevations['zeta_max']
    LOGGER.info(f'sample size: {samples.shape}')

    if storm_name is not None:
        storm = BestTrackForcing(storm_name)
    else:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')

    # calculate the distance of each node to the storm track
    storm_linestring = LineString(list(zip(storm.data['longitude'], storm.data['latitude'])))
    nodes = GeoDataFrame(
        samples['node'],
        index=samples['node'],
        geometry=geopandas.points_from_xy(samples['x'], samples['y']),
    )
    samples = samples.assign_coords(
        {'distance_to_track': ('node', nodes.distance(storm_linestring))}
    )
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

    if plot_results:
        LOGGER.info(f'running surrogate on {samples.shape} samples')
        surrogate_mean = chaospy.E(poly=surrogate_model, dist=distribution)
        surrogate_std = chaospy.Std(poly=surrogate_model, dist=distribution)
        modeled_mean = samples.mean('run')
        modeled_std = samples.std('run')

        surrogate_mean = xarray.DataArray(
            surrogate_mean, coords=modeled_mean.coords, dims=modeled_mean.dims,
        )
        surrogate_std = xarray.DataArray(
            surrogate_std, coords=modeled_std.coords, dims=modeled_std.dims,
        )

        node_results = xarray.Dataset(
            {
                'mean': xarray.combine_nested(
                    [surrogate_mean, modeled_mean], concat_dim='source'
                ).assign_coords({'source': ['surrogate', 'model']}),
                'std': xarray.combine_nested(
                    [surrogate_std, modeled_std], concat_dim='source'
                ).assign_coords({'source': ['surrogate', 'model']}),
                'difference': xarray.ufuncs.fabs(surrogate_std - modeled_std),
            }
        )

        plot_nodes_across_runs(
            node_results,
            title=f'surrogate-predicted and modeled elevations for {len(node_results["node"])} nodes',
            node_colors='mean',
            storm=storm,
            output_filename=input_directory / 'elevations.png' if save_plots else None,
        )

    if plot_percentile:
        percentiles = [10, 50, 90]
        percentile_filename = input_directory / 'percentiles.nc'
        if not percentile_filename.exists():
            surrogate_percentiles = get_percentiles(
                samples=samples,
                percentiles=percentiles,
                surrogate_model=surrogate_model,
                distribution=distribution,
            )

            modeled_percentiles = samples.quantile(
                dim='run', q=surrogate_percentiles['quantile'] / 100
            )
            modeled_percentiles.coords['quantile'] = surrogate_percentiles['quantile']

            node_percentiles = xarray.combine_nested(
                [surrogate_percentiles, modeled_percentiles], concat_dim='source'
            ).assign_coords(source=['surrogate', 'model'])

            node_percentiles = node_percentiles.to_dataset(name='quantiles')

            node_percentiles.assign(
                difference=xarray.ufuncs.fabs(surrogate_percentiles - modeled_percentiles)
            )

            LOGGER.info(f'saving percentiles to "{percentile_filename}"')
            node_percentiles.to_netcdf(percentile_filename)
        else:
            LOGGER.info(f'loading percentiles from "{percentile_filename}"')
            node_percentiles = xarray.open_dataset(percentile_filename)

        node_percentiles = xarray.Dataset(
            {
                str(float(percentile.values)): node_percentiles['quantiles'].sel(
                    quantile=percentile
                )
                for percentile in node_percentiles['quantile']
            },
            coords=node_percentiles.coords,
        )

        plot_nodes_across_runs(
            node_percentiles,
            title=f'{len(percentiles)} surrogate-predicted percentile(s) for {len(node_percentiles["node"])} nodes',
            node_colors='90.0',
            storm=storm,
            output_filename=input_directory / 'percentiles.png' if save_plots else None,
        )

    if show_plots:
        pyplot.show()
