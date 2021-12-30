from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import dask
from matplotlib import pyplot
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
    percentiles_from_surrogate,
    sensitivities_from_surrogate,
    statistics_from_surrogate,
    surrogate_from_training_set,
    validations_from_surrogate,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_nodes')

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
                elevation_selection=True,
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
                xarray.ufuncs.logical_and(
                    ~elevations['zeta'].isnull().any('time').any('run'),  # only wet nodes
                    FieldOutput.subset(
                        elevations['node'], maximum_depth=0, bounds=subset_bounds
                    ),
                ),
                drop=True,
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

    surrogate_model = surrogate_from_training_set(
        training_set=training_set,
        training_perturbations=training_perturbations,
        distribution=distribution,
        filename=surrogate_filename,
        use_quadrature=use_quadrature,
    )

    if make_sensitivities_plot:
        sensitivities = sensitivities_from_surrogate(
            surrogate_model=surrogate_model,
            distribution=distribution,
            variables=perturbations['variable'],
            nodes=subset,
            filename=sensitivities_filename,
        )
        plot_sensitivities(
            sensitivities=sensitivities,
            storm=storm,
            output_filename=input_directory / 'sensitivities.png' if save_plots else None,
        )

    if make_validation_plot:
        node_validation = validations_from_surrogate(
            surrogate_model=surrogate_model,
            training_set=training_set,
            training_perturbations=training_perturbations,
            validation_set=validation_set,
            validation_perturbations=validation_perturbations,
            filename=validation_filename,
        )

        plot_validations(
            validation=node_validation,
            output_filename=input_directory / 'validation.png' if save_plots else None,
        )

    if make_statistics_plot:
        node_statistics = statistics_from_surrogate(
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
            logarithmic=True,
        )

    if make_percentile_plot:
        percentiles = [10, 50, 90]
        node_percentiles = percentiles_from_surrogate(
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
            logarithmic=True,
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
            logarithmic=True,
        )

    if show_plots:
        LOGGER.info('showing plots')
        pyplot.show()
