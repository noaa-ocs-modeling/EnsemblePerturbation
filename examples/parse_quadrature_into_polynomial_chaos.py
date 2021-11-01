import math
from os import PathLike
from pathlib import Path

from adcircpy.forcing import BestTrackForcing
from adcircpy.forcing.winds.best_track import VortexForcing
import cartopy
import chaospy
import dask
import geopandas
from matplotlib import cm, colors, gridspec, pyplot
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import numpy
import pyproj
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs, FieldOutput
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.uncertainty_quantification.quadrature import (
    fit_surrogate_to_quadrature,
    get_percentiles,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_nodes')


def plot_node_map(
    nodes: xarray.Dataset,
    map_title: str = None,
    node_colors: [(float, float, float)] = None,
    storm: str = None,
    map_axis: pyplot.Axes = None,
):
    if map_title is None:
        map_title = f'{len(nodes["node"])} nodes'

    map_crs = cartopy.crs.PlateCarree()
    if map_axis is None:
        map_axis = pyplot.Axes(projection=map_crs)

    map_bounds = [
        float(nodes.coords['x'].min().values),
        float(nodes.coords['y'].min().values),
        float(nodes.coords['x'].max().values),
        float(nodes.coords['y'].max().values),
    ]

    countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    if storm is None:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')
    elif not isinstance(storm, VortexForcing):
        storm = BestTrackForcing(storm)

    if node_colors is None:
        color_map = cm.get_cmap('jet')
        color_values = numpy.arange(len(nodes['node']))
        normalization = colors.Normalize(
            vmin=numpy.min(color_values), vmax=numpy.max(color_values)
        )
        color_values = normalization(color_values)
        node_colors = color_map(color_values)
    elif isinstance(node_colors, str):
        color_map = cm.get_cmap('jet')
        map_title = f'"{node_colors}" of {map_title}'
        color_values = nodes[node_colors]
        if len(color_values.dims) > 1:
            color_values = color_values.mean(
                [dim for dim in color_values.dims if dim != 'node']
            )
        min_value = numpy.min(color_values.values)
        max_value = numpy.max(color_values.values)
        try:
            normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
        except ValueError:
            normalization = colors.Normalize(vmin=min_value, vmax=max_value)
        color_values = normalization(color_values)
        node_colors = color_map(color_values)
    else:
        color_map = cm.get_cmap('jet')
        min_value = numpy.min(node_colors.values)
        max_value = numpy.max(node_colors.values)
        try:
            normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
        except ValueError:
            normalization = colors.Normalize(vmin=min_value, vmax=max_value)
        color_values = normalization(node_colors)
        node_colors = color_map(color_values)

    countries.plot(color='lightgrey', ax=map_axis)
    storm.data.plot(
        x='longitude',
        y='latitude',
        ax=map_axis,
        label=storm_name,
        legend=storm_name is not None,
    )

    map_axis.scatter(
        x=nodes['x'], y=nodes['y'], c=node_colors, s=2, norm=normalization, transform=map_crs
    )

    map_axis.set_xlim(map_bounds[0], map_bounds[2])
    map_axis.set_ylim(map_bounds[1], map_bounds[3])
    map_axis.set_title(map_title)


def plot_nodes_across_runs(
    nodes: xarray.Dataset,
    title: str = None,
    node_colors: [(float, float, float)] = None,
    storm: str = None,
    output_filename: PathLike = None,
):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    if title is not None:
        figure.suptitle(title)

    grid = gridspec.GridSpec(len(nodes.data_vars), 2, figure=figure)

    map_crs = cartopy.crs.PlateCarree()
    map_axis = figure.add_subplot(grid[:, 0], projection=map_crs)

    if node_colors is None:
        color_map = cm.get_cmap('jet')
        color_values = numpy.arange(len(nodes['node']))
        normalization = colors.Normalize(
            vmin=numpy.min(color_values), vmax=numpy.max(color_values)
        )
        color_values = normalization(color_values)
        node_colors = color_map(color_values)
    elif isinstance(node_colors, str):
        color_map = cm.get_cmap('jet')
        color_values = nodes[node_colors]
        if len(color_values.dims) > 1:
            color_values = color_values.mean(
                [dim for dim in color_values.dims if dim != 'node']
            )
        min_value = numpy.min(color_values.values)
        max_value = numpy.max(color_values.values)
        try:
            normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
            colorbar = figure.colorbar(
                mappable=cm.ScalarMappable(cmap=color_map, norm=normalization), ax=map_axis,
            )
        except ValueError:
            normalization = colors.Normalize(vmin=min_value, vmax=max_value)
            colorbar = figure.colorbar(
                mappable=cm.ScalarMappable(cmap=color_map, norm=normalization), ax=map_axis,
            )
        colorbar.set_label(node_colors)
        color_values = normalization(color_values)
        node_colors = color_map(color_values)
    else:
        color_map = cm.get_cmap('jet')
        min_value = numpy.min(node_colors.values)
        max_value = numpy.max(node_colors.values)
        try:
            normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
        except ValueError:
            normalization = colors.Normalize(vmin=min_value, vmax=max_value)
        color_values = normalization(node_colors)
        node_colors = color_map(color_values)

    plot_node_map(nodes, node_colors=node_colors, storm=storm, map_axis=map_axis)

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

        if 'source' in variable.dims:
            sources = variable['source']
        else:
            sources = [None]

        for source_index, source in enumerate(sources):
            if source is None:
                color_map = cm.get_cmap('jet')
                variable_node_colors = color_map(color_values)
            elif source == 'model':
                color_map = cm.get_cmap('jet')
                variable_node_colors = color_map(color_values)
            elif source == 'surrogate':
                variable_node_colors = 'grey'

            kwargs = {}
            if source == 'surrogate':
                kwargs['linestyle'] = '--'

            if 'source' in variable.dims:
                source_data = variable.sel(source=source)
            else:
                source_data = variable

            if 'time' in nodes.dims:
                for node_index in range(len(nodes['node'])):
                    node_data = source_data.isel(node=node_index)
                    node_color = variable_node_colors[node_index]
                    node_data.plot.line(
                        x='time', c=node_color, ax=variable_axis, **kwargs,
                    )
                    if variable_name == 'mean' and 'std' in nodes.data_vars:
                        std_data = nodes['std'].isel(node=node_index)
                        if 'source' in std_data.dims:
                            std_data = std_data.sel(source=source)
                        variable_axis.fill_between(
                            training_set['time'],
                            node_data - std_data,
                            node_data + std_data,
                            color=node_color,
                            alpha=0.3,
                            **kwargs,
                        )
            else:
                bar_width = 0.01
                bar_offset = bar_width * (source_index + 0.5 - len(sources) / 2)

                variable_axis.bar(
                    x=source_data['distance_to_track'] + bar_offset,
                    width=bar_width,
                    height=source_data.values,
                    color=variable_node_colors,
                    **kwargs,
                )

        variable_axis.set_title(variable_name)
        variable_axis.tick_params(axis='x', which='both', labelsize=6)
        variable_axis.set(xlabel=None)
        # variable_axis.set_yscale('symlog')

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def comparison_plot_grid(variables: [str], figure: Figure = None):
    if figure is None:
        figure = pyplot.figure()

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

    if len(shared_grid_columns['y']) == 0:
        shared_grid_columns['y'][variables[-1]] = None

    axes = {}
    for row_index in range(grid_length):
        row_variable = list(shared_grid_columns['y'])[row_index]
        axes[row_variable] = {}
        for column_index in range(grid_length - row_index):
            column_variable = list(shared_grid_columns['x'])[column_index]

            sharex = shared_grid_columns['x'][column_variable]
            sharey = shared_grid_columns['y'][row_variable]

            variable_axis = figure.add_subplot(
                grid[row_index, column_index], sharex=sharex, sharey=sharey,
            )

            if sharex is None:
                shared_grid_columns['x'][column_variable] = variable_axis
            if sharey is None:
                shared_grid_columns['y'][row_variable] = variable_axis

            if grid_length != 1:
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
            else:
                variable_axis.set_xlabel(column_variable)
                variable_axis.set_ylabel(row_variable)

            axes[row_variable][column_variable] = variable_axis

    return axes, grid


def plot_perturbed_variables(
    perturbations: xarray.Dataset, title: str = None, output_filename: PathLike = None,
):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    if title is None:
        title = f'{len(perturbations["run"])} pertubation(s) of {len(perturbations["variable"])} variable(s)'
    figure.suptitle(title)

    variables = perturbations['variable'].values
    axes, grid = comparison_plot_grid(variables, figure=figure)

    color_map = cm.get_cmap('jet')

    perturbation_colors = perturbations['weights']
    if not perturbation_colors.isnull().values.all():
        color_map_axis = figure.add_subplot(grid[-1, -1])
        color_map_axis.set_visible(False)
        min_value = numpy.min(perturbation_colors.values)
        max_value = numpy.max(perturbation_colors.values)
        perturbation_colors.loc[perturbation_colors.isnull()] = 0
        try:
            normalization = colors.LogNorm(vmin=min_value, vmax=max_value)
            colorbar = figure.colorbar(
                mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
                orientation='horizontal',
                ax=color_map_axis,
            )
        except ValueError:
            normalization = colors.Normalize(vmin=min_value, vmax=max_value)
            colorbar = figure.colorbar(
                mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
                orientation='horizontal',
                ax=color_map_axis,
            )
        colorbar.set_label('weight')
    else:
        perturbation_colors = numpy.arange(len(perturbation_colors))
        normalization = None

    perturbations = perturbations['perturbations']
    for row_variable, columns in axes.items():
        for column_variable, axis in columns.items():
            axis.scatter(
                perturbations.sel(variable=column_variable),
                perturbations.sel(variable=row_variable),
                c=perturbation_colors,
                cmap=color_map,
                norm=normalization,
            )

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def plot_comparison(
    nodes: xarray.DataArray, title: str = None, output_filename: Axis = None,
):
    if 'source' not in nodes.dims:
        raise ValueError(f'"source" not found in data array dimensions: {nodes.dims}')

    sources = nodes['source'].values

    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    if title is not None:
        title = f'comparison of {len(sources)} sources along {len(nodes["node"])} node(s)'
    figure.suptitle(title)

    axes, grid = comparison_plot_grid(sources, figure=figure)

    for row_source, columns in axes.items():
        for column_source, axis in columns.items():
            axis.scatter(
                nodes.sel(source=column_source),
                nodes.sel(source=row_source),
                c=perturbations['weights'],
            )

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    plot_perturbations = True
    plot_validation = True
    plot_statistics = True
    plot_percentile = True

    save_plots = True
    show_plots = False

    storm_name = None

    input_directory = Path.cwd()
    surrogate_filename = input_directory / 'surrogate.npy'
    validation_filename = input_directory / 'validation.nc'

    filenames = ['perturbations.nc', 'maxele.63.nc']

    datasets = {}
    existing_filenames = []
    for filename in filenames:
        filename = input_directory / filename
        if filename.exists():
            datasets[filename.name] = xarray.open_dataset(filename, chunks={})
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
    max_elevations = datasets['maxele.63.nc']

    training_perturbations = perturbations.sel(
        run=perturbations['run'].str.contains('quadrature')
    )
    validation_perturbations = perturbations.drop_sel(run=training_perturbations['run'])

    if plot_perturbations:
        plot_perturbed_variables(
            training_perturbations,
            title=f'{len(training_perturbations["run"])} training pertubation(s) of {len(training_perturbations["variable"])} variable(s)',
            output_filename=input_directory / 'training_perturbations.png',
        )
        plot_perturbed_variables(
            validation_perturbations,
            title=f'{len(validation_perturbations["run"])} validation pertubation(s) of {len(validation_perturbations["variable"])} variable(s)',
            output_filename=input_directory / 'validation_perturbations.png',
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
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        samples = max_elevations['zeta_max'].drop_sel(run='original')
        samples = samples.sel(node=FieldOutput.subset(samples['node'], bounds=subset_bounds))

    # calculate the distance of each node to the storm track
    if storm_name is not None:
        storm = BestTrackForcing(storm_name)
    else:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')
    geoid = pyproj.Geod(ellps='WGS84')
    nodes = numpy.stack([samples['x'], samples['y']], axis=1)
    storm_points = storm.data[['longitude', 'latitude']].values
    distances = numpy.fromiter(
        (
            numpy.min(
                geoid.inv(
                    *numpy.repeat(
                        numpy.expand_dims(node, axis=0), repeats=len(storm_points), axis=0
                    ).T,
                    *storm_points.T,
                )[-1]
            )
            for node in nodes
        ),
        dtype=float,
        count=len(samples['node']),
    )
    samples = samples.assign_coords({'distance_to_track': ('node', distances)})
    samples = samples.sortby('distance_to_track')

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        training_set = samples.sel(run=training_perturbations['run'])
        validation_set = samples.sel(run=validation_perturbations['run'])

    LOGGER.info(f'total {training_set.shape} training samples')
    LOGGER.info(f'total {validation_set.shape} validation samples')

    if not surrogate_filename.exists():
        # expand polynomials with polynomial chaos
        polynomials = chaospy.generate_expansion(
            order=3, dist=distribution, rule='three_terms_recurrence',
        )

        surrogate_model = fit_surrogate_to_quadrature(
            samples=training_set,
            polynomials=polynomials,
            perturbations=training_perturbations['perturbations'],
            weights=training_perturbations['weights'],
        )

        with open(surrogate_filename, 'wb') as surrogate_file:
            LOGGER.info(f'saving surrogate model to "{surrogate_filename}"')
            surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{surrogate_filename}"')
        surrogate_model = chaospy.load(surrogate_filename, allow_pickle=True)

    if plot_validation:
        if not validation_filename.exists():
            LOGGER.info(f'running surrogate model on {training_set.shape} training samples')
            training_results = surrogate_model(*training_perturbations['perturbations'].T).T
            training_results = numpy.stack([training_set, training_results], axis=0)
            training_results = xarray.DataArray(
                training_results,
                coords={'source': ['model', 'surrogate'], **training_set.coords},
                dims=('source', 'run', 'node'),
                name='training',
            )

            LOGGER.info(
                f'running surrogate model on {validation_set.shape} validation samples'
            )
            validation_results = surrogate_model(
                *validation_perturbations['perturbations'].T
            ).T
            validation_results = numpy.stack([validation_set, validation_results], axis=0)
            validation_results = xarray.DataArray(
                validation_results,
                coords={'source': ['model', 'surrogate'], **validation_set.coords},
                dims=('source', 'run', 'node'),
                name='validation',
            )

            validation_results = xarray.combine_nested(
                [training_results, validation_results], concat_dim='type'
            )
            validation_results = validation_results.assign_coords(
                type=['training', 'validation']
            )
            validation_results = validation_results.to_dataset(name='results')

            LOGGER.info(f'saving validation to "{validation_filename}"')
            validation_results.to_netcdf(validation_filename)
        else:
            LOGGER.info(f'loading validation from "{validation_filename}"')
            validation_results = xarray.open_dataset(validation_filename)

        validation_results = validation_results['results']

        sources = validation_results['source'].values

        figure = pyplot.figure()
        figure.set_size_inches(12, 12 / 1.61803398875)
        figure.suptitle(
            f'comparison of {len(sources)} sources along {len(validation_results["node"])} node(s)'
        )

        axes, grid = comparison_plot_grid(sources, figure=figure)

        type_colors = {'training': 'b', 'validation': 'r'}

        for row_source, columns in axes.items():
            for column_source, axis in columns.items():
                for result_type in validation_results['type'].values:
                    type_results = validation_results.sel(type=result_type)
                    axis.scatter(
                        type_results.sel(source=column_source),
                        type_results.sel(source=row_source),
                        c=type_colors[result_type],
                        label=result_type,
                    )
                max_value = max(axis.get_xlim()[-1], axis.get_ylim()[-1])
                axis.plot([0, max_value], [0, max_value], '--k')
                axis.legend()

        output_filename = input_directory / 'validation.png' if save_plots else None
        if output_filename is not None:
            figure.savefig(output_filename, dpi=200, bbox_inches='tight')

    if plot_statistics:
        LOGGER.info(
            f'gathering mean and standard deviation from surrogate on {training_set.shape} training samples'
        )
        surrogate_mean = chaospy.E(poly=surrogate_model, dist=distribution)
        surrogate_std = chaospy.Std(poly=surrogate_model, dist=distribution)
        modeled_mean = training_set.mean('run')
        modeled_std = training_set.std('run')

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
            title=f'surrogate-predicted and modeled elevation(s) for {len(node_results["node"])} node(s) across {len(training_set["run"])} run(s)',
            node_colors='mean',
            storm=storm,
            output_filename=input_directory / 'elevations.png' if save_plots else None,
        )

    if plot_percentile:
        percentiles = [10, 50, 90]
        percentile_filename = input_directory / 'percentiles.nc'
        if not percentile_filename.exists():
            surrogate_percentiles = get_percentiles(
                samples=training_set,
                percentiles=percentiles,
                surrogate_model=surrogate_model,
                distribution=distribution,
            )

            modeled_percentiles = training_set.quantile(
                dim='run', q=surrogate_percentiles['quantile'] / 100
            )
            modeled_percentiles.coords['quantile'] = surrogate_percentiles['quantile']

            node_percentiles = xarray.combine_nested(
                [surrogate_percentiles, modeled_percentiles], concat_dim='source'
            ).assign_coords(source=['surrogate', 'model'])

            node_percentiles = node_percentiles.to_dataset(name='quantiles')

            node_percentiles = node_percentiles.assign(
                differences=xarray.ufuncs.fabs(surrogate_percentiles - modeled_percentiles)
            )

            LOGGER.info(f'saving percentiles to "{percentile_filename}"')
            node_percentiles.to_netcdf(percentile_filename)
        else:
            LOGGER.info(f'loading percentiles from "{percentile_filename}"')
            node_percentiles = xarray.open_dataset(percentile_filename)

        plot_nodes_across_runs(
            xarray.Dataset(
                {
                    str(float(percentile.values)): node_percentiles['quantiles'].sel(
                        quantile=percentile
                    )
                    for percentile in node_percentiles['quantile']
                },
                coords=node_percentiles.coords,
            ),
            title=f'{len(percentiles)} surrogate-predicted and modeled percentile(s) for {len(node_percentiles["node"])} node(s) across {len(training_set["run"])} run(s)',
            node_colors='90.0',
            storm=storm,
            output_filename=input_directory / 'percentiles.png' if save_plots else None,
        )

        plot_nodes_across_runs(
            xarray.Dataset(
                {
                    str(float(percentile.values)): node_percentiles['differences'].sel(
                        quantile=percentile
                    )
                    for percentile in node_percentiles['quantile']
                },
                coords={
                    coord_name: coord
                    for coord_name, coord in node_percentiles.coords.items()
                    if coord_name != 'source'
                },
            ),
            title=f'differences between {len(percentiles)} surrogate-predicted and modeled percentile(s) for {len(node_percentiles["node"])} node(s) across {len(training_set["run"])} run(s)',
            node_colors='90.0',
            storm=storm,
            output_filename=input_directory / 'percentile_differences.png'
            if save_plots
            else None,
        )

    if show_plots:
        LOGGER.info('showing plots')
        pyplot.show()
