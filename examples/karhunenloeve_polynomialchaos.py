from os import PathLike
from pathlib import Path
from typing import List

from adcircpy.forcing import BestTrackForcing
import chaospy
import dask
from matplotlib import pyplot
import numpoly
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
    fit_surrogate,
    get_percentiles_from_surrogate,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parse_nodes')


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

    LOGGER.info(f'Evaluating Karhunen-Loeve expansion from {ngrid} grid nodes and {nens} ensemble members')

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


def get_surrogate_model(
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    distribution: chaospy.Distribution,
    filename: PathLike,
    use_quadrature: bool = False,
    polynomial_order: int = 3,
) -> numpoly.ndpoly:
    """
    use ``chaospy`` to build a surrogate model from the given training set / perturbations and single / joint distribution

    :param training_set: array of data along nodes in the mesh to use to fit the model
    :param training_perturbations: perturbations along each variable space that comprise the cloud of model inputs
    :param distribution: ``chaospy`` distribution
    :param filename: path to file to store polynomial
    :param use_quadrature: assume that the variable perturbations and training set are built along a quadrature, and fit accordingly
    :param polynomial_order: order of the polynomial chaos expansion
    :return: polynomial
    """

    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
        # expand polynomials with polynomial chaos
        polynomial_expansion = chaospy.generate_expansion(
            order=polynomial_order, dist=distribution, rule='three_terms_recurrence',
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
    surrogate_model: numpoly.ndpoly,
    distribution: chaospy.Distribution,
    variables: [str],
    nodes: xarray.Dataset,
    filename: PathLike,
) -> xarray.DataArray:
    """
    Get sensitivities of a given order for the surrogate model and distribution.

    :param surrogate_model: polynomial representing the surrogate model
    :param distribution: single or joint distribution of variable space
    :param variables: variable names
    :param nodes: dataset containing node information (nodes and XYZ coordinates) of mesh
    :param filename: filename to store sensitivities
    :return: array of sensitivities per node per variable
    """

    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
        LOGGER.info(f'extracting sensitivities from surrogate model and distribution')

        sensitivities = [
            chaospy.Sens_t(surrogate_model, distribution),
            chaospy.Sens_m(surrogate_model, distribution),
        ]

        sensitivities = numpy.stack(sensitivities)

        sensitivities = xarray.DataArray(
            sensitivities,
            coords={
                'order': ['total', 'main'],
                'variable': variables,
                'node': nodes['node'],
                'x': nodes['x'],
                'y': nodes['y'],
                'depth': nodes['depth'],
            },
            dims=('order', 'variable', 'node'),
        ).T

        sensitivities = sensitivities.to_dataset(name='sensitivities')

        LOGGER.info(f'saving sensitivities to "{filename}"')
        sensitivities.to_netcdf(filename)
    else:
        LOGGER.info(f'loading sensitivities from "{filename}"')
        sensitivities = xarray.open_dataset(filename)

    return sensitivities['sensitivities']


def get_validations(
    surrogate_model: numpoly.ndpoly,
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    validation_set: xarray.Dataset,
    validation_perturbations: xarray.Dataset,
    filename: PathLike,
) -> xarray.Dataset:
    """


    :param surrogate_model: polynomial of surrogate model to query
    :param training_set: set of training data (across nodes and perturbations)
    :param training_perturbations: array of perturbations corresponding to training set
    :param validation_set: set of validation data (across nodes and perturbations)
    :param validation_perturbations: array of perturbations corresponding to validation set
    :param filename: file path to which to save
    :return: array of validations
    """

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
            [training_results.drop('type'), node_validation.drop('type')], concat_dim='type'
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
    surrogate_model: numpoly.ndpoly,
    distribution: chaospy.Distribution,
    training_set: xarray.Dataset,
    filename: PathLike,
) -> xarray.Dataset:
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
    percentiles: List[float],
    surrogate_model: numpoly.ndpoly,
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

    surrogate_model = get_surrogate_model(
        training_set=training_set,
        training_perturbations=training_perturbations,
        distribution=distribution,
        filename=surrogate_filename,
        use_quadrature=use_quadrature,
        polynomial_order=polynomial_order,
    )

    if make_sensitivities_plot:
        sensitivities = get_sensitivities(
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
        node_validation = get_validations(
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
