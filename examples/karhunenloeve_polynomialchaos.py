from pathlib import Path
import pickle

from adcircpy.forcing import BestTrackForcing
import chaospy
import dask
from matplotlib import pyplot
import numpy
import xarray

from ensembleperturbation.parsing.adcirc import FieldOutput
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.plotting import (
    plot_nodes_across_runs,
    plot_perturbations,
    plot_sensitivities,
    plot_validations,
    plot_selected_validations,
    plot_selected_percentiles,
    plot_kl_surrogate_fit,
)
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import (
    karhunen_loeve_expansion,
    karhunen_loeve_prediction,
)
from ensembleperturbation.uncertainty_quantification.surrogate import (
    percentiles_from_surrogate,
    sensitivities_from_surrogate,
    statistics_from_surrogate,
    surrogate_from_karhunen_loeve,
    surrogate_from_training_set,
    validations_from_surrogate,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_karhunen_loeve')

if __name__ == '__main__':
    # PC parameters
    use_quadrature = True
    polynomial_order = 3
    # KL parameters
    variance_explained = 0.99
    # subsetting parameters
    subset_bounds = (-81, 32, -75, 37)
    depth_bounds = 25.0
    point_spacing = 10
    # analysis type
    #use_depth = True   # for depths (must be >= 0)
    use_depth = False # for elevations

    make_perturbations_plot = True
    make_klprediction_plot = True
    make_klsurrogate_plot = True
    make_sensitivities_plot = True
    make_validation_plot = True
    make_percentile_plot = True

    save_plots = True
    show_plots = False

    storm_name = None

    input_directory = Path.cwd()
    if use_depth:
        output_directory = input_directory / 'outputs_depths'
    else:
        output_directory = input_directory / 'outputs_elevations'
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    subset_filename = output_directory / 'subset.nc'
    kl_filename = output_directory / 'karhunen_loeve.pkl'
    kl_surrogate_filename = output_directory / 'kl_surrogate.npy'
    surrogate_filename = output_directory / 'surrogate.npy'
    kl_validation_filename = output_directory / 'kl_surrogate_fit.nc'
    sensitivities_filename = output_directory / 'sensitivities.nc'
    validation_filename = output_directory / 'validation.nc'
    statistics_filename = output_directory / 'statistics.nc'
    percentile_filename = output_directory / 'percentiles.nc'

    filenames = ['perturbations.nc', 'maxele.63.nc']

    datasets = {}
    existing_filenames = []
    for filename in filenames:
        filename = input_directory / filename
        if filename.exists():
            datasets[filename.name] = xarray.open_dataset(filename, chunks='auto')
        else:
            raise FileNotFoundError(filename.name)

    perturbations = datasets['perturbations.nc']
    max_elevations = datasets['maxele.63.nc']

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
            output_directory=output_directory if save_plots else None,
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

    # sample based on subset and always wet locations
    values = max_elevations['zeta_max']
    if not subset_filename.exists():
        LOGGER.info('subsetting nodes')
        num_nodes = len(values['node'])
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            subsetted_nodes = values['node'].where(
                numpy.logical_and(
                    ~values.isnull().any('run'),
                    FieldOutput.subset(
                        values['node'], maximum_depth=depth_bounds, bounds=subset_bounds
                    ),
                ),
                drop=True,
            )
            subsetted_nodes = subsetted_nodes[::point_spacing]

            subset = values.drop_sel(run='original')
            subset = subset.sel(node=subsetted_nodes)
        subset = subset.chunk({'node': -1})
        if len(subset['node']) != num_nodes:
            LOGGER.info(
                f'subsetted down to {len(subset["node"])} nodes ({len(subset["node"]) / num_nodes:.1%})'
            )
        LOGGER.info(f'saving subset to "{subset_filename}"')
        subset.to_netcdf(subset_filename)
    else:
        LOGGER.info(f'loading subset from "{subset_filename}"')
        subset = xarray.open_dataset(subset_filename)[values.name]

    if storm_name is not None:
        storm = BestTrackForcing(storm_name)
    else:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        training_set = subset.sel(run=training_perturbations['run']) + subset['depth'] * use_depth
        validation_set = subset.sel(run=validation_perturbations['run']) + subset['depth'] * use_depth

    LOGGER.info(f'total {training_set.shape} training samples')
    LOGGER.info(f'total {validation_set.shape} validation samples')

    # Evaluating the Karhunen-Loeve expansion
    nens, ngrid = training_set.shape
    if not kl_filename.exists():
        LOGGER.info(
            f'Evaluating Karhunen-Loeve expansion from {ngrid} grid nodes and {nens} ensemble members'
        )
        kl_expansion = karhunen_loeve_expansion(
            training_set.values.T, neig=variance_explained, output_directory=output_directory,
        )
    else:
        LOGGER.info(f'loading Karhunen-Loeve expansion from "{kl_filename}"')
        with open(kl_filename, 'rb') as kl_handle:
            kl_expansion = pickle.load(kl_handle)

    neig = len(kl_expansion['eigenvalues'])  # number of eigenvalues
    LOGGER.info(f'found {neig} Karhunen-Loeve modes')
    LOGGER.info(f'Karhunen-Loeve expansion: {list(kl_expansion)}')

    # plot prediction versus actual simulated
    if make_klprediction_plot:
        kl_predicted = karhunen_loeve_prediction(
            kl_dict=kl_expansion,
            actual_values=training_set.T,
            ensembles_to_plot=[0, int(nens / 2), nens - 1],
            plot_directory=output_directory,
        )

    # evaluate the surrogate for each KL sample
    kl_training_set = xarray.DataArray(data=kl_expansion['samples'], dims=['run', 'mode'])
    kl_surrogate_model = surrogate_from_training_set(
        training_set=kl_training_set,
        training_perturbations=training_perturbations,
        distribution=distribution,
        filename=kl_surrogate_filename,
        use_quadrature=use_quadrature,
        polynomial_order=polynomial_order,
    )

    # plot kl surrogate model versus training set
    if make_klsurrogate_plot:
        kl_fit = validations_from_surrogate(
            surrogate_model=kl_surrogate_model,
            training_set=kl_training_set,
            training_perturbations=training_perturbations,
            filename=kl_validation_filename,
        )
        
        plot_kl_surrogate_fit(
            kl_fit=kl_fit,
            output_filename=output_directory / 'kl_surrogate_fit.png' if save_plots else None,
        )

    # convert the KL surrogate model to the overall surrogate at each node
    surrogate_model = surrogate_from_karhunen_loeve(
        mean_vector=kl_expansion['mean_vector'],
        eigenvalues=kl_expansion['eigenvalues'],
        modes=kl_expansion['modes'],
        kl_surrogate_model=kl_surrogate_model,
        filename=surrogate_filename,
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
            output_filename=output_directory / 'sensitivities.png' if save_plots else None,
        )

    if make_validation_plot:
        node_validation = validations_from_surrogate(
            surrogate_model=surrogate_model,
            training_set=training_set,
            training_perturbations=training_perturbations,
            validation_set=validation_set,
            validation_perturbations=validation_perturbations,
            enforce_positivity=use_depth,
            filename=validation_filename,
        )

        plot_validations(
            validation=node_validation,
            output_filename=output_directory / 'validation.png' if save_plots else None,
        )
     
        plot_selected_validations(
            validation=node_validation,
            run_list=node_validation['run'][numpy.arange(0,50,9)].values,
            output_directory=output_directory if save_plots else None,
        )

    if make_percentile_plot:
        percentiles = [10, 50, 90]
        node_percentiles = percentiles_from_surrogate(
            surrogate_model=surrogate_model,
            distribution=distribution,
            training_set=validation_set,
            percentiles=percentiles,
            enforce_positivity=use_depth,
            filename=percentile_filename,
        )

        plot_selected_percentiles(
            node_percentiles=node_percentiles,
            perc_list=percentiles,
            output_directory=output_directory if save_plots else None,
        )

    if show_plots:
        LOGGER.info('showing plots')
        pyplot.show()
