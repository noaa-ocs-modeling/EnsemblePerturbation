from os import PathLike
from pathlib import Path
from typing import List

import chaospy
import numpoly
import numpy
from sklearn.linear_model import OrthogonalMatchingPursuit
import xarray

from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('surrogate')


def surrogate_from_karhunen_loeve(
    kl_dict: dict,
    kl_surrogate_model: numpoly.ndpoly,
    filename: PathLike = None,
) -> numpoly.ndpoly:
    """
    Get the joint Karhunen-Loeve / Polynomial Chaos polynomial
    from Eq (6) in Sargsyan, K. and Ricciuto D. (2021):
    sum_{k=0}^{K} (kroneckerdelta_{k0}*mean(f(z)) +  
                   sum_{j=0}^{L} [c_{jk}*\sqrt(ev_j)*ef_j(z)]) 
    K -> num PC coefficients
    L -> num KL modes
    c -> PC coefficient 
    ev -> eigenvalue
    ef -> eigenfunction
    """
    
    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    num_points = len(kl_dict['mean_vector'])
    num_modes = len(kl_dict['eigenvalues'])
    assert (
        num_modes == len(kl_surrogate_model)
    ), 'number of kl_dict eigenvalues must be equal to the length of the kl_surrogate_model'
    
    if filename is None or not filename.exists():
        # get the coefficients of the PC for each point in z (spatiotemporal dimension)
        pc_exponents = kl_surrogate_model.exponents
        pc_coefficients = numpy.array(kl_surrogate_model.coefficients)
        num_coefficients = pc_coefficients.shape[0]
        klpc_coefficients = numpy.zeros((num_coefficients, num_points))
        klpc_coefficients[0, :] = kl_dict['mean_vector']
        for z_index in range(num_points):
            for coef_index in range(num_coefficients):
                for mode_index in range(num_modes):
                    klpc_coefficients[coef_index, z_index] += (
                        pc_coefficients[coef_index, mode_index]
                        * numpy.sqrt(kl_dict['eigenvalues'][mode_index])
                        * kl_dict['modes'][z_index, mode_index]
                    )
        surrogate_model = numpoly.ndpoly.from_attributes(
            exponents=pc_exponents, coefficients=klpc_coefficients,
        )
        
        if filename is not None:
            with open(filename, 'wb') as surrogate_file:
                LOGGER.info(f'saving surrogate model to "{filename}"')
                surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{filename}"')
        surrogate_model = chaospy.load(filename, allow_pickle=True)

    return surrogate_model
    

def surrogate_from_samples(
    samples: xarray.DataArray,
    perturbations: xarray.DataArray,
    polynomials: numpoly.ndpoly,
    quadrature: bool = False,
    quadrature_weights: xarray.DataArray = None,
) -> numpoly.ndpoly:
    # create surrogate models for selected nodes
    if quadrature:
        LOGGER.info(
            f'fitting polynomial surrogate to {samples.shape} samples along the quadrature'
        )
        if quadrature_weights is None:
            LOGGER.warning('no quadrature weights provided')
        try:
            surrogate_model = chaospy.fit_quadrature(
                orth=polynomials,
                nodes=perturbations.T,
                weights=quadrature_weights,
                solves=samples,
            )
        except AssertionError:
            if (
                perturbations.shape[0] != len(quadrature_weights)
                or len(quadrature_weights) != len(samples)
                or len(samples) != perturbations.shape[0]
            ):
                raise AssertionError(
                    f'{perturbations.shape[0]} != {len(quadrature_weights)} != {len(samples)}'
                )
            else:
                raise
    else:
        LOGGER.info(
            f'fitting polynomial surrogate to {samples.shape} samples using regression'
        )
        try:
            model = OrthogonalMatchingPursuit(n_nonzero_coefs=3, fit_intercept=False)
            surrogate_model = chaospy.fit_regression(
                polynomials=polynomials, abscissas=perturbations.T, evals=samples, model=model,
            )
        except AssertionError:
            if perturbations.T.shape[-1] != len(samples):
                raise AssertionError(f'{perturbations.T.shape[-1]} != {len(samples)}')
            else:
                raise

    return surrogate_model


def surrogate_from_training_set(
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    distribution: chaospy.Distribution,
    filename: PathLike = None,
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

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
        # expand polynomials with polynomial chaos
        polynomial_expansion = chaospy.generate_expansion(
            order=polynomial_order, dist=distribution, rule='three_terms_recurrence',
        )

        if not use_quadrature:
            training_shape = training_set.shape
            training_set = training_set.sel(node=~training_set.isnull().any('run'))
            if training_set.shape != training_shape:
                LOGGER.info(f'dropped `NaN`s to {training_set.shape}')

        surrogate_model = surrogate_from_samples(
            samples=training_set,
            perturbations=training_perturbations['perturbations'],
            polynomials=polynomial_expansion,
            quadrature=use_quadrature,
            quadrature_weights=training_perturbations['weights'] if use_quadrature else None,
        )

        if filename is not None:
            with open(filename, 'wb') as surrogate_file:
                LOGGER.info(f'saving surrogate model to "{filename}"')
                surrogate_model.dump(surrogate_file)
    else:
        LOGGER.info(f'loading surrogate model from "{filename}"')
        surrogate_model = chaospy.load(filename, allow_pickle=True)

    return surrogate_model


def sensitivities_from_surrogate(
    surrogate_model: numpoly.ndpoly,
    distribution: chaospy.Distribution,
    variables: [str],
    nodes: xarray.Dataset,
    filename: PathLike = None,
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

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
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

        if filename is not None:
            LOGGER.info(f'saving sensitivities to "{filename}"')
            sensitivities.to_netcdf(filename)
    else:
        LOGGER.info(f'loading sensitivities from "{filename}"')
        sensitivities = xarray.open_dataset(filename)

    return sensitivities['sensitivities']


def validations_from_surrogate(
    surrogate_model: numpoly.ndpoly,
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    validation_set: xarray.Dataset,
    validation_perturbations: xarray.Dataset,
    filename: PathLike = None,
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

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
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

        if filename is not None:
            LOGGER.info(f'saving validation to "{filename}"')
            node_validation.to_netcdf(filename)
    else:
        LOGGER.info(f'loading validation from "{filename}"')
        node_validation = xarray.open_dataset(filename)

    return node_validation


def statistics_from_surrogate(
    surrogate_model: numpoly.ndpoly,
    distribution: chaospy.Distribution,
    training_set: xarray.Dataset,
    filename: PathLike = None,
) -> xarray.Dataset:
    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
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
                'difference': numpy.fabs(surrogate_std - modeled_std),
            }
        )

        if filename is not None:
            LOGGER.info(f'saving statistics to "{filename}"')
            node_statistics.to_netcdf(filename)
    else:
        LOGGER.info(f'loading statistics from "{filename}"')
        node_statistics = xarray.open_dataset(filename)

    return node_statistics


def percentiles_from_samples(
    samples: xarray.DataArray,
    percentiles: List[float],
    surrogate_model: numpoly.ndpoly,
    distribution: chaospy.Distribution,
) -> xarray.DataArray:
    LOGGER.info(f'calculating {len(percentiles)} percentile(s): {percentiles}')
    surrogate_percentiles = chaospy.Perc(
        poly=surrogate_model, q=percentiles, dist=distribution, sample=samples.shape[1],
    )

    surrogate_percentiles = xarray.DataArray(
        surrogate_percentiles,
        coords={
            'quantile': percentiles,
            **{
                coord: values
                for coord, values in samples.coords.items()
                if coord not in ['run', 'type']
            },
        },
        dims=('quantile', *(dim for dim in samples.dims if dim not in ['run', 'type'])),
    )

    return surrogate_percentiles


def percentiles_from_surrogate(
    percentiles: List[float],
    surrogate_model: numpoly.ndpoly,
    distribution: chaospy.Distribution,
    training_set: xarray.Dataset,
    filename: PathLike = None,
) -> xarray.Dataset:
    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
        surrogate_percentiles = percentiles_from_samples(
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
            differences=numpy.fabs(surrogate_percentiles - modeled_percentiles)
        )

        if filename is not None:
            LOGGER.info(f'saving percentiles to "{filename}"')
            node_percentiles.to_netcdf(filename)
    else:
        LOGGER.info(f'loading percentiles from "{filename}"')
        node_percentiles = xarray.open_dataset(filename)

    return node_percentiles
