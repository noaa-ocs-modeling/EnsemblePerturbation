from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import dask
import geopandas
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from modelforcings.vortex import VortexForcing
import numpy
import pyproj
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs, FieldOutput
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.plotting import (
    plot_comparison,
    plot_nodes_across_runs,
    plot_perturbed_variables,
)
from ensembleperturbation.uncertainty_quantification.surrogate import (
    fit_surrogate,
    get_percentiles,
)
from ensembleperturbation.utilities import encode_categorical_values, get_logger

LOGGER = get_logger('parse_nodes')

if __name__ == '__main__':
    use_quadrature = True

    plot_perturbations = True
    plot_sensitivities = True
    plot_validation = True
    plot_statistics = True
    plot_percentile = True

    save_plots = True
    show_plots = False

    storm_name = None

    input_directory = Path.cwd()
    subset_filename = input_directory / 'subset.nc'
    surrogate_filename = input_directory / 'surrogate.npy'
    sensitivities_filename = input_directory / 'sensitivities.nc'
    validation_filename = input_directory / 'validation.nc'
    statistics_filename = input_directory / 'statistics.nc'
    percentile_filename = input_directory / 'percentiles.nc'

    filenames = ['perturbations.nc', 'maxele.63.nc', 'fort.63.nc']

    datasets = {}
    existing_filenames = []
    for filename in filenames:
        filename = input_directory / filename
        if filename.exists():
            datasets[filename.name] = xarray.open_dataset(filename, chunks='auto')
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
    elevations = datasets['fort.63.nc']

    perturbations = perturbations.assign_coords(
        type=(
            'run',
            (
                numpy.where(
                    perturbations['run'].str.contains('quadrature'), 'training', 'validation'
                )
            ),
        )
    )

    training_perturbations = perturbations.sel(run=perturbations['type'] == 'training')
    validation_perturbations = perturbations.sel(run=perturbations['type'] == 'validation')

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

        track_directory = input_directory / 'track_files'
        if track_directory.exists():
            track_filenames = {
                track_filename.stem: track_filename
                for track_filename in track_directory.glob('*.22')
            }

            figure = pyplot.figure()
            figure.set_size_inches(12, 12 / 1.61803398875)
            figure.suptitle(f'{len(track_filenames)} perturbations of storm')

            map_axis = figure.add_subplot(1, 1, 1)
            countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

            runs = perturbations['run'].values
            perturbation_types = perturbations['type'].values
            unique_perturbation_types = numpy.unique(perturbation_types)
            encoded_perturbation_types = encode_categorical_values(
                perturbation_types, unique_values=unique_perturbation_types
            )
            linear_normalization = Normalize()
            colors = get_cmap('jet')(linear_normalization(encoded_perturbation_types))

            bounds = numpy.array([None, None, None, None])
            for index, run in enumerate(runs):
                storm = VortexForcing.from_fort22(track_filenames[run]).data
                points = storm.loc[:, ['longitude', 'latitude']].values.reshape(-1, 1, 2)
                segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(
                    segments,
                    linewidths=numpy.concatenate(
                        [
                            [0],
                            storm['radius_of_maximum_winds']
                            / max(storm['radius_of_maximum_winds']),
                        ]
                    )
                    * 4,
                    color=colors[index],
                )
                map_axis.add_collection(line_collection)

                track_bounds = numpy.array(
                    [
                        points[:, :, 0].min(),
                        points[:, :, 1].min(),
                        points[:, :, 0].max(),
                        points[:, :, 1].max(),
                    ]
                )
                if bounds[0] is None or track_bounds[0] < bounds[0]:
                    bounds[0] = track_bounds[0]
                if bounds[1] is None or track_bounds[1] < bounds[1]:
                    bounds[1] = track_bounds[1]
                if bounds[2] is None or track_bounds[2] > bounds[2]:
                    bounds[2] = track_bounds[2]
                if bounds[3] is None or track_bounds[3] > bounds[3]:
                    bounds[3] = track_bounds[3]

            map_axis.set_xlim((bounds[0], bounds[2]))
            map_axis.set_ylim((bounds[1], bounds[3]))

            unique_perturbation_type_colors = get_cmap('jet')(
                linear_normalization(numpy.unique(encoded_perturbation_types))
            )
            map_axis.legend(
                [Patch(facecolor=color) for color in unique_perturbation_type_colors],
                unique_perturbation_types,
            )

            xlim = map_axis.get_xlim()
            ylim = map_axis.get_ylim()

            countries.plot(color='lightgrey', ax=map_axis)

            map_axis.set_xlim(xlim)
            map_axis.set_ylim(ylim)

            if save_plots:
                figure.savefig(
                    input_directory / 'storm_tracks.png', dpi=200, bbox_inches='tight'
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
    values = max_elevations['zeta_max']
    subset_bounds = (-83, 25, -72, 42)
    if not subset_filename.exists():
        LOGGER.info('subsetting nodes')
        num_nodes = len(values['node'])
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            subsetted_nodes = elevations['node'].where(
                FieldOutput.subset(elevations['node'], bounds=subset_bounds), drop=True,
            )
            subset = values.drop_sel(run='original')
            subset = subset.sel(node=subsetted_nodes)
        if len(subset['node']) != num_nodes:
            LOGGER.info(
                f'subsetted down to {len(subset["node"])} nodes ({len(subset["node"]) / num_nodes:.1%})'
            )
        LOGGER.info(f'saving subset to "{subset_filename}"')
        subset.to_netcdf(subset_filename)
    else:
        LOGGER.info(f'loading subset from "{subset_filename}"')
        subset = xarray.open_dataset(subset_filename)[values.name]

    # calculate the distance of each node to the storm track
    if storm_name is not None:
        storm = BestTrackForcing(storm_name)
    else:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')
    geoid = pyproj.Geod(ellps='WGS84')
    nodes = numpy.stack([subset['x'], subset['y']], axis=1)
    storm_points = storm.data[['longitude', 'latitude']].values
    distances = numpy.fromiter(
        (
            geoid.inv(
                *numpy.repeat(
                    numpy.expand_dims(node, axis=0), repeats=len(storm_points), axis=0
                ).T,
                *storm_points.T,
            )[-1].min()
            for node in nodes
        ),
        dtype=float,
        count=len(subset['node']),
    )
    subset = subset.assign_coords({'distance_to_track': ('node', distances)})
    subset = subset.sortby('distance_to_track')

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        training_set = subset.sel(run=training_perturbations['run'])
        validation_set = subset.sel(run=validation_perturbations['run'])

    LOGGER.info(f'total {training_set.shape} training samples')
    LOGGER.info(f'total {validation_set.shape} validation samples')

    if not surrogate_filename.exists():
        # expand polynomials with polynomial chaos
        polynomial_expansion = chaospy.generate_expansion(
            order=3, dist=distribution, rule='three_terms_recurrence',
        )

        if not use_quadrature:
            training_shape = training_set.shape
            training_set = training_set.sel(node=~training_set.isnull().any('run'))
            if training_set.shape != training_shape:
                LOGGER.info(f'dropped `NaN`s to {training_set.shape}')

        surrogate_model = fit_surrogate(
            samples=training_set,
            perturbations=training_perturbations['perturbations'],
            polynomials=polynomial_expansion,
            quadrature=use_quadrature,
            quadrature_weights=training_perturbations['weights'] if use_quadrature else None,
        )

        with open(surrogate_filename, 'wb') as surrogate_file:
            LOGGER.info(f'saving surrogate model to "{surrogate_filename}"')
            surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{surrogate_filename}"')
        surrogate_model = chaospy.load(surrogate_filename, allow_pickle=True)

    if plot_sensitivities:
        if not sensitivities_filename.exists():
            LOGGER.info(f'extracting sensitivities from surrogate model and distribution')
            sensitivities = chaospy.Sens_m(surrogate_model, distribution)
            sensitivities = xarray.DataArray(
                sensitivities,
                coords={'variable': perturbations['variable'], 'node': subset['node']},
                dims=('variable', 'node'),
            ).T

            sensitivities = sensitivities.to_dataset(name='sensitivities')

            LOGGER.info(f'saving sensitivities to "{sensitivities_filename}"')
            sensitivities.to_netcdf(sensitivities_filename)
        else:
            LOGGER.info(f'loading sensitivities from "{sensitivities_filename}"')
            sensitivities = xarray.open_dataset(sensitivities_filename)['sensitivities']

        figure = pyplot.figure()
        figure.set_size_inches(12, 12 / 1.61803398875)
        figure.suptitle(
            f'Sobol sensitivities of {len(sensitivities["variable"])} variables along {len(sensitivities["node"])} node(s)'
        )

        axis = figure.add_subplot(1, 1, 1)

        for variable in sensitivities['variable']:
            axis.scatter(
                sensitivities['node'], sensitivities.sel(variable=variable), label=variable
            )

        if save_plots:
            figure.savefig(input_directory / 'sensitivities.png', dpi=200, bbox_inches='tight')

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
            node_validation = surrogate_model(*validation_perturbations['perturbations'].T).T
            node_validation = numpy.stack([validation_set, node_validation], axis=0)
            node_validation = xarray.DataArray(
                node_validation,
                coords={'source': ['model', 'surrogate'], **validation_set.coords},
                dims=('source', 'run', 'node'),
                name='validation',
            )

            node_validation = xarray.combine_nested(
                [training_results, node_validation], concat_dim='type'
            )
            node_validation = node_validation.assign_coords(type=['training', 'validation'])
            node_validation = node_validation.to_dataset(name='results')

            LOGGER.info(f'saving validation to "{validation_filename}"')
            node_validation.to_netcdf(validation_filename)
        else:
            LOGGER.info(f'loading validation from "{validation_filename}"')
            node_validation = xarray.open_dataset(validation_filename)

        node_validation = node_validation['results']

        sources = node_validation['source'].values

        figure = pyplot.figure()
        figure.set_size_inches(12, 12 / 1.61803398875)
        figure.suptitle(
            f'comparison of {len(sources)} sources along {len(node_validation["node"])} node(s)'
        )

        type_colors = {'training': 'b', 'validation': 'r'}
        axes = None
        for index, result_type in enumerate(node_validation['type'].values):
            result_validation = node_validation.sel(type=result_type)
            axes = plot_comparison(
                result_validation,
                title=f'comparison of {len(sources)} sources along {len(result_validation["node"])} node(s)',
                reference_line=index == 0,
                figure=figure,
                axes=axes,
                s=1,
                c=type_colors[result_type],
                label=result_type,
            )

        for row in axes.values():
            for axis in row.values():
                axis.legend()

        if save_plots:
            figure.savefig(input_directory / 'validation.png', dpi=200, bbox_inches='tight')

    if plot_statistics:
        if not statistics_filename.exists():
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

            node_statistics = xarray.Dataset(
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

            LOGGER.info(f'saving statistics to "{statistics_filename}"')
            node_statistics.to_netcdf(statistics_filename)
        else:
            LOGGER.info(f'loading statistics from "{statistics_filename}"')
            node_statistics = xarray.open_dataset(statistics_filename)

        plot_nodes_across_runs(
            node_statistics,
            title=f'surrogate-predicted and modeled elevation(s) for {len(node_statistics["node"])} node(s) across {len(training_set["run"])} run(s)',
            colors='mean',
            storm=storm,
            output_filename=input_directory / 'elevations.png' if save_plots else None,
        )

    if plot_percentile:
        percentiles = [10, 50, 90]
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
            colors='90.0',
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
            colors='90.0',
            storm=storm,
            output_filename=input_directory / 'percentile_differences.png'
            if save_plots
            else None,
        )

    if show_plots:
        LOGGER.info('showing plots')
        pyplot.show()
