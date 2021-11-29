from os import PathLike
from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import dask
from matplotlib import pyplot
import numpoly
import numpy
import pyproj
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs, FieldOutput
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.plotting import (
    plot_nodes_across_runs,
    plot_perturbations,
    plot_sensitivities,
    plot_validations,
)
from ensembleperturbation.uncertainty_quantification.surrogate import (
    fit_surrogate,
    get_percentiles_from_surrogate,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_nodes')


def get_surrogate_model(
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    distribution: chaospy.Distribution,
    filename: PathLike,
    use_quadrature: bool = False,
) -> numpoly.PolyLike:
    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
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

        with open(filename, 'wb') as surrogate_file:
            LOGGER.info(f'saving surrogate model to "{filename}"')
            surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{filename}"')
        surrogate_model = chaospy.load(filename, allow_pickle=True)

    return surrogate_model


def get_sensitivities(
    surrogate_model: numpoly.PolyLike,
    distribution: chaospy.Distribution,
    perturbations: xarray.Dataset,
    subset: xarray.Dataset,
    filename: PathLike,
) -> xarray.Dataset:
    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
        LOGGER.info(f'extracting sensitivities from surrogate model and distribution')
        sensitivities = chaospy.Sens_m(surrogate_model, distribution)
        sensitivities = xarray.DataArray(
            sensitivities,
            coords={'variable': perturbations['variable'], 'node': subset['node']},
            dims=('variable', 'node'),
        ).T

        sensitivities = sensitivities.to_dataset(name='sensitivities')

        LOGGER.info(f'saving sensitivities to "{filename}"')
        sensitivities.to_netcdf(filename)
    else:
        LOGGER.info(f'loading sensitivities from "{filename}"')
        sensitivities = xarray.open_dataset(filename)['sensitivities']

    return sensitivities


def get_validations(
    surrogate_model: numpoly.PolyLike,
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    validation_set: xarray.Dataset,
    filename: PathLike,
) -> xarray.Dataset:
    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
        LOGGER.info(f'running surrogate model on {training_set.shape} training samples')
        training_results = surrogate_model(*training_perturbations['perturbations'].T).T
        training_results = numpy.stack([training_set, training_results], axis=0)
        training_results = xarray.DataArray(
            training_results,
            coords={'source': ['model', 'surrogate'], **training_set.coords},
            dims=('source', 'run', 'node'),
            name='training',
        )

        LOGGER.info(f'running surrogate model on {validation_set.shape} validation samples')
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

        LOGGER.info(f'saving validation to "{filename}"')
        node_validation.to_netcdf(filename)
    else:
        LOGGER.info(f'loading validation from "{filename}"')
        node_validation = xarray.open_dataset(filename)

    return node_validation


def get_statistics(
    surrogate_model: numpoly.PolyLike,
    distribution: chaospy.Distribution,
    training_set: xarray.Dataset,
    filename: PathLike,
):
    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
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

        LOGGER.info(f'saving statistics to "{filename}"')
        node_statistics.to_netcdf(filename)
    else:
        LOGGER.info(f'loading statistics from "{filename}"')
        node_statistics = xarray.open_dataset(filename)

    return node_statistics


def get_percentiles(
    percentiles: [float],
    surrogate_model: numpoly.PolyLike,
    distribution: chaospy.Distribution,
    training_set: xarray.Dataset,
    filename: PathLike,
) -> xarray.Dataset:
    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
        surrogate_percentiles = get_percentiles_from_surrogate(
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

        LOGGER.info(f'saving percentiles to "{filename}"')
        node_percentiles.to_netcdf(filename)
    else:
        LOGGER.info(f'loading percentiles from "{filename}"')
        node_percentiles = xarray.open_dataset(filename)

    return node_percentiles


if __name__ == '__main__':
    use_quadrature = True

    make_perturbations_plot = True
    make_sensitivities_plot = True
    make_validation_plot = True
    make_statistics_plot = True
    make_percentile_plot = True

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

    if make_perturbations_plot:
        plot_perturbations(
            training_perturbations=training_perturbations,
            validation_perturbations=validation_perturbations,
            runs=perturbations['run'].values,
            perturbation_types=perturbations['type'].values,
            track_directory=input_directory / 'track_files',
            output_directory=input_directory if save_plots else None,
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
    # TODO: sample based on sensitivity / eigenvalues
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

    surrogate_model = get_surrogate_model(
        training_set=training_set,
        training_perturbations=training_perturbations,
        distribution=distribution,
        filename=surrogate_filename,
        use_quadrature=use_quadrature,
    )

    if make_sensitivities_plot:
        sensitivities = get_sensitivities(
            surrogate_model=surrogate_model,
            distribution=distribution,
            perturbations=perturbations,
            subset=subset,
            filename=sensitivities_filename,
        )
        plot_sensitivities(
            sensitivities=sensitivities,
            output_filename=input_directory / 'sensitivities.png' if save_plots else None,
        )

    if make_validation_plot:
        node_validation = get_validations(
            surrogate_model=surrogate_model,
            training_set=training_set,
            training_perturbations=training_perturbations,
            validation_set=validation_set,
            filename=validation_filename,
        )

        plot_validations(
            validation=node_validation,
            output_filename=input_directory / 'validation.png' if save_plots else None,
        )

    if make_statistics_plot:
        node_statistics = get_statistics(
            surrogate_model=surrogate_model,
            distribution=distribution,
            training_set=training_set,
            filename=statistics_filename,
        )

        plot_nodes_across_runs(
            node_statistics,
            title=f'surrogate-predicted and modeled elevation(s) for {len(node_statistics["node"])} node(s) across {len(training_set["run"])} run(s)',
            colors='mean',
            storm=storm,
            output_filename=input_directory / 'elevations.png' if save_plots else None,
        )

    if make_percentile_plot:
        percentiles = [10, 50, 90]
        node_percentiles = get_percentiles(
            surrogate_model=surrogate_model,
            distribution=distribution,
            training_set=training_set,
            percentiles=percentiles,
            filename=percentile_filename,
        )

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
