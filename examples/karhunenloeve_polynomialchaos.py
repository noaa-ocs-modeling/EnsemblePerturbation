from os import PathLike
from pathlib import Path

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
)
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import (
    karhunen_loeve_expansion,
)
from ensembleperturbation.uncertainty_quantification.surrogate import (
    percentiles_from_surrogate,
    sensitivities_from_surrogate,
    statistics_from_surrogate,
    surrogate_from_training_set,
    validations_from_surrogate,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_karhunen_loeve')


def get_karhunenloeve_expansion(
    training_set: xarray.Dataset, filename: PathLike, variance_explained: float = 0.95,
) -> dict:
    """
    use karhunen_loeve_expansion class of EnsemblePerturbation to build the Karhunen-Loeve expansion

    :param training_set: array of data along nodes in the mesh to use to fit the model
    :param filename: path to file to store Karhunen-Loeve eigenvalues/eignenvectors as a dictionary 
    :param variance_explained: the cutoff for the variance explained by the KL expansion, so that the number of eigenvalues retained is reduced
    :return: kl_dict
    """

    ymodel = training_set.T
    ngrid = ymodel.shape[0]  # number of points
    nens = ymodel.shape[1]  # number of ensembles

    LOGGER.info(
        f'Evaluating Karhunen-Loeve expansion from {ngrid} grid nodes and {nens} ensemble members'
    )

    ## Evaluating the KL mode
    # Components of the dictionary:
    # mean_vector is the average field                                        : size (ngrid,)
    # modes is the KL modes ('principal directions')                          : size (ngrid,neig)
    # eigenvalues is the eigenvalue vector                                    : size (neig,)
    # samples are the samples for the KL coefficients                         : size (nens, neig)
    kl_dict = karhunen_loeve_expansion(ymodel, neig=variance_explained, plot=False)

    neig = len(kl_dict['eigenvalues'])  # number of eigenvalues
    LOGGER.info(f'found {neig} Karhunen-Loeve modes')

    # # evaluate the fit of the KL prediction
    # # ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
    # ypred = karhunen_loeve_prediction(kl_dict)

    # plot scatter points to compare ymodel and ypred spatially
    # for example in numpy.linspace(0, nens, num=10, endpoint=False, dtype=int):
    #    # plot_coastline()
    #    plot_points(
    #        np.hstack((points_subset, ymodel[:, [example]])),
    #        save_filename='modeled_zmax' + str(example),
    #        title='modeled zmax, ensemble #' + str(example),
    #        vmax=3.0,
    #        vmin=0.0,
    #    )
    #
    #    # plot_coastline()
    #    plot_points(
    #        np.hstack((points_subset, ypred[:, [example]])),
    #        save_filename='predicted_zmax' + str(example),
    #        title='predicted zmax, ensemble #' + str(example),
    #        vmax=3.0,
    #        vmin=0.0,
    #    )

    return kl_dict


if __name__ == '__main__':
    use_quadrature = True
    polynomial_order = 3
    variance_explained = 0.95

    make_perturbations_plot = False
    make_sensitivities_plot = False
    make_validation_plot = False
    make_statistics_plot = False
    make_percentile_plot = False

    save_plots = True
    show_plots = False

    storm_name = None

    input_directory = Path.cwd()
    subset_filename = input_directory / 'subset.nc'
    kl_filename = input_directory / 'karhunen_loeve.npy'
    surrogate_filename = input_directory / 'surrogate.npy'
    sensitivities_filename = input_directory / 'sensitivities.nc'
    validation_filename = input_directory / 'validation.nc'
    statistics_filename = input_directory / 'statistics.nc'
    percentile_filename = input_directory / 'percentiles.nc'

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

    # sample based on subset and always wet locations
    values = max_elevations['zeta_max']
    subset_bounds = (-81, 32, -76, 36.5)
    if not subset_filename.exists():
        LOGGER.info('subsetting nodes')
        num_nodes = len(values['node'])
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            subsetted_nodes = values['node'].where(
                xarray.ufuncs.logical_and(
                    ~values.isnull().any('run')['node'],
                    FieldOutput.subset(values['node'], bounds=subset_bounds),
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

    if storm_name is not None:
        storm = BestTrackForcing(storm_name)
    else:
        storm = BestTrackForcing.from_fort22(input_directory / 'track_files' / 'original.22')

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        training_set = subset.sel(run=training_perturbations['run'])
        validation_set = subset.sel(run=validation_perturbations['run'])

    LOGGER.info(f'total {training_set.shape} training samples')
    LOGGER.info(f'total {validation_set.shape} validation samples')

    # Evaluating the Karhunen-Loeve expansion
    kl_expansion = get_karhunenloeve_expansion(
        training_set=training_set, variance_explained=variance_explained, filename=kl_filename,
    )

    LOGGER.info(f'Karhunen-Loeve expansion: {kl_expansion}')

    surrogate_model = surrogate_from_training_set(
        training_set=training_set,
        training_perturbations=training_perturbations,
        distribution=distribution,
        filename=surrogate_filename,
        use_quadrature=use_quadrature,
        polynomial_order=polynomial_order,
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
