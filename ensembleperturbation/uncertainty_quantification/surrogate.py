from os import PathLike
from pathlib import Path
from typing import List, Union

import chaospy
import numpoly
import numpy
import sklearn
import xarray

from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('surrogate')


def surrogate_from_karhunen_loeve(
    mean_vector: numpy.ndarray,
    eigenvalues: numpy.ndarray,
    modes: numpy.ndarray,
    kl_surrogate_model: numpoly.ndpoly,
    filename: PathLike = None,
) -> numpoly.ndpoly:
    """
    build a polynomial from the given Karhunen-Loeve expansion (eigenvalues and modes) along with a mean vector along the node space

    the joint Karhunen-Loeve / Polynomial Chaos polynomial is derived from Eq (6) in Sargsyan, K. and Ricciuto D. (2021):

    .. math::

        \sum_{k=0}^{K} ( \delta_{k0} * \overline{f(z)} + \sum_{j=0}^{L} [ c_{jk} * \sqrt{ev_j} * ef_j(z) ] )

    ``K`` -> num PC coefficients

    ``L`` -> num KL modes

    ``c`` -> PC coefficient

    ``ev`` -> eigenvalue

    ``ef`` -> eigenfunction

    :param mean_vector: all points
    :param eigenvalues: eigenvalues of each point to each mode
    :param modes: modes of the Karhunen-Loeve expansion
    :param kl_surrogate_model: ``ndpoly`` surrogate polynomial generated on training set
    :param filename: file path to save surrogate as pickled ``ndpoly`` object
    :return: polynomial constructued from the above equation
    """

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    num_points = len(mean_vector)
    num_modes = len(eigenvalues)

    assert num_modes == len(
        kl_surrogate_model
    ), 'number of kl_dict eigenvalues must be equal to the length of the kl_surrogate_model'

    LOGGER.info(f'transforming surrogate to {num_points} points from {num_modes} eigenmodes')
    if filename is None or not filename.exists():
        # get the coefficients of the PC for each point in z (spatiotemporal dimension)
        pc_exponents = kl_surrogate_model.exponents
        pc_coefficients = numpy.array(kl_surrogate_model.coefficients)
        klpc_coefficients = numpy.sum(
            numpy.stack(
                [
                    numpy.dot(
                        (pc_coefficients * numpy.sqrt(eigenvalues))[:, mode_index, None],
                        modes[None, mode_index, :],
                    )
                    for mode_index in range(num_modes)
                ],
                axis=0,
            ),
            axis=0,
        )
        klpc_coefficients[0, :] += mean_vector

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
    regression_model: sklearn.linear_model,
    norms: numpy.ndarray = None,
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
                norms=norms,
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
            poly_list = [None] * samples.shape[1]
            for mode in range(samples.shape[1]):
                poly_list[mode] = chaospy.fit_regression(
                    polynomials=polynomials,
                    abscissas=perturbations.T,
                    evals=samples[:, mode],
                    model=regression_model,
                )
            surrogate_model = numpoly.polynomial(poly_list)
            ## Or just call this for default regression
            # surrogate_model = chaospy.fit_regression(polynomials=polynomials,abscissas=perturbations.T,evals=samples)
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
    regression_model: sklearn.linear_model = sklearn.linear_model.LinearRegression(
        fit_intercept=False
    ),
) -> numpoly.ndpoly:
    """
    use ``chaospy`` to build a surrogate model from the given training set / perturbations and single / joint distribution

    :param training_set: array of data along nodes in the mesh to use to fit the model
    :param training_perturbations: perturbations along each variable space that comprise the cloud of model inputs
    :param distribution: ``chaospy`` distribution
    :param filename: path to file to store polynomial
    :param use_quadrature: assume that the variable perturbations and training set are built along a quadrature, and fit accordingly
    :param polynomial_order: order of the polynomial chaos expansion
    :param regression_model: the scikit linear model to use to fit the polynomial regression [LinearRegression (OLS) by default]: https://scikit-learn.org/stable/modules/linear_model.html
    :return: polynomial
    """

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
        # expand polynomials with polynomial chaos
        polynomial_expansion, norms = chaospy.generate_expansion(
            order=polynomial_order,
            dist=distribution,
            rule='three_terms_recurrence',
            retall=True,
        )

        surrogate_model = surrogate_from_samples(
            samples=training_set,
            perturbations=training_perturbations['perturbations'],
            polynomials=polynomial_expansion,
            norms=norms,
            quadrature=use_quadrature,
            quadrature_weights=training_perturbations['weights'] if use_quadrature else None,
            regression_model=regression_model,
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
    element_table: xarray.DataArray = None,
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
            chaospy.Sens_m(surrogate_model, distribution),
            chaospy.Sens_t(surrogate_model, distribution),
        ]

        sensitivities = numpy.stack(sensitivities)

        sensitivities = xarray.DataArray(
            sensitivities,
            coords={
                'order': ['main', 'total'],
                'variable': variables,
                'node': nodes['node'],
                'x': nodes['x'],
                'y': nodes['y'],
                'depth': nodes['depth'],
            },
            dims=('order', 'variable', 'node'),
        ).T

        sensitivities = sensitivities.to_dataset(name='sensitivities')

        if element_table is not None:
            sensitivities = sensitivities.assign_coords({'element': element_table})

        if filename is not None:
            LOGGER.info(f'saving sensitivities to "{filename}"')
            sensitivities.to_netcdf(filename)
    else:
        LOGGER.info(f'loading sensitivities from "{filename}"')
        sensitivities = xarray.open_dataset(filename)

    return sensitivities


def validations_from_surrogate(
    surrogate_model: numpoly.ndpoly,
    training_set: xarray.Dataset,
    training_perturbations: xarray.Dataset,
    validation_set: xarray.Dataset = None,
    validation_perturbations: xarray.Dataset = None,
    minimum_allowable_value: float = None,
    convert_from_log_scale: bool = False,
    convert_from_depths: Union[bool, float] = False,
    element_table: xarray.DataArray = None,
    filename: PathLike = None,
) -> xarray.Dataset:
    """


    :param surrogate_model: polynomial of surrogate model to query
    :param training_set: set of training data (across nodes and perturbations)
    :param training_perturbations: array of perturbations corresponding to training set
    :param validation_set: set of validation data (across nodes and perturbations)
    :param validation_perturbations: array of perturbations corresponding to validation set
    :param minimum_allowable_value: if surrogate prediction falls below this value set to NaN
    :param convert_from_log_scale: whether to take the exp() of the result
    :param convert_from_depths: whether to substract still water depth from the result
    :param filename: file path to which to save
    :return: array of validations
    """

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
        LOGGER.info(f'running surrogate model on {training_set.shape} training samples')
        training_results = surrogate_model(*training_perturbations['perturbations'].T).T
        if convert_from_log_scale:
            training_results = numpy.exp(training_results)
        if isinstance(convert_from_depths, (float, numpy.ndarray)):
            training_results -= convert_from_depths
        if minimum_allowable_value is not None:
            # compare to adjusted depths if provided
            too_small = training_results < minimum_allowable_value
            training_results[too_small] = numpy.nan
        if bool(convert_from_depths):
            training_results -= training_set['depth'].values

        training_results = numpy.stack([training_set, training_results], axis=0)
        training_results = xarray.DataArray(
            training_results,
            coords={'source': ['model', 'surrogate'], **training_set.coords},
            dims=('source', 'run', 'node'),
            name='training',
        )

        if validation_set is None:
            node_validation = training_results.to_dataset(name='results')
        else:
            LOGGER.info(
                f'running surrogate model on {validation_set.shape} validation samples'
            )
            node_validation = surrogate_model(*validation_perturbations['perturbations'].T).T
            if convert_from_log_scale:
                node_validation = numpy.exp(node_validation)
            if isinstance(convert_from_depths, (float, numpy.ndarray)):
                node_validation -= convert_from_depths
            if minimum_allowable_value is not None:
                too_small = node_validation < minimum_allowable_value
                node_validation[too_small] = numpy.nan
            if bool(convert_from_depths):
                node_validation -= validation_set['depth'].values

            node_validation = numpy.stack([validation_set, node_validation], axis=0)
            node_validation = xarray.DataArray(
                node_validation,
                coords={'source': ['model', 'surrogate'], **validation_set.coords},
                dims=('source', 'run', 'node'),
                name='validation',
            )

            node_validation = xarray.combine_nested(
                [training_results.drop('type'), node_validation.drop('type')],
                concat_dim='type',
            )
            node_validation = node_validation.assign_coords(type=['training', 'validation'])
            node_validation = node_validation.to_dataset(name='results')

        if element_table is not None:
            node_validation = node_validation.assign_coords({'element': element_table})

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
    convert_from_log_scale: bool = False,
) -> xarray.DataArray:
    LOGGER.info(f'calculating {len(percentiles)} percentile(s): {percentiles}')
    # surrogate_percentiles = chaospy.Perc(
    #    poly=surrogate_model, q=percentiles, dist=distribution, sample=samples.shape[1],
    # )
    surrogate_percentiles = compute_surrogate_percentiles(
        poly=surrogate_model,
        q=percentiles,
        dist=distribution,
        convert_from_log_scale=convert_from_log_scale,
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
    minimum_allowable_value: float = None,
    convert_from_log_scale: bool = False,
    convert_from_depths: Union[bool, float] = False,
    element_table: xarray.DataArray = None,
    filename: PathLike = None,
) -> xarray.Dataset:
    """


    :param percentiles: positions where percentiles are taken. Must be a number or a list, 
        where all values are on the interval ``[0, 100]``.
    :param surrogate_model: polynomial of surrogate model to query
    :parama distribution: surrogate model ``chaospy`` distribution model
    :param training_set: set of training data (across nodes and perturbations)
    :param minimum_allowable_value: if surrogate prediction falls below this value set to NaN
    :param convert_from_log_scale: whether to take the exp() of the result
    :param convert_from_depths: whether to substract still water depth from the result
    :param filename: file path to which to save
    :return: array of percentiles
    """

    if filename is not None and not isinstance(filename, Path):
        filename = Path(filename)

    if filename is None or not filename.exists():
        surrogate_percentiles = percentiles_from_samples(
            samples=training_set,
            percentiles=percentiles,
            surrogate_model=surrogate_model,
            distribution=distribution,
            convert_from_log_scale=convert_from_log_scale,
        )

        # before evaluating quantile for model set null water elevation to the ground elevation
        training_set = numpy.fmax(training_set, -training_set['depth'])
        modeled_percentiles = training_set.quantile(
            dim='run', q=surrogate_percentiles['quantile'] / 100
        )

        if isinstance(convert_from_depths, (float, numpy.ndarray)):
            surrogate_percentiles -= convert_from_depths
        if minimum_allowable_value is not None:
            too_small = (
                modeled_percentiles + training_set['depth']
            ).values < minimum_allowable_value
            modeled_percentiles.values[too_small] = numpy.nan
            too_small = surrogate_percentiles.values < minimum_allowable_value
            surrogate_percentiles.values[too_small] = numpy.nan
        if bool(convert_from_depths):
            surrogate_percentiles -= training_set['depth']

        modeled_percentiles.coords['quantile'] = surrogate_percentiles['quantile']

        node_percentiles = xarray.combine_nested(
            [surrogate_percentiles, modeled_percentiles], concat_dim='source'
        ).assign_coords(source=['surrogate', 'model'])

        node_percentiles = node_percentiles.to_dataset(name='quantiles')

        node_percentiles = node_percentiles.assign(
            differences=numpy.fabs(surrogate_percentiles - modeled_percentiles)
        )

        if element_table is not None:
            node_percentiles = node_percentiles.assign_coords({'element': element_table})

        if filename is not None:
            LOGGER.info(f'saving percentiles to "{filename}"')
            node_percentiles.to_netcdf(filename)
    else:
        LOGGER.info(f'loading percentiles from "{filename}"')
        node_percentiles = xarray.open_dataset(filename)

    return node_percentiles


def compute_surrogate_percentiles(
    poly: numpoly.ndpoly,
    q: List[float],
    dist: chaospy.Distribution,
    sample: int = 10000,
    convert_from_log_scale: bool = False,
    **kws,
):
    """
    Percentile function 
    *Modified to be able to convert from log scale

    Note that this function is an empirical function that operates using Monte
    Carlo sampling.

    Args:
        poly (numpoly.ndpoly):
            Polynomial of interest.
        q (numpy.ndarray):
            positions where percentiles are taken. Must be a number or an
            array, where all values are on the interval ``[0, 100]``.
        dist (Distribution):
            Defines the space where percentile is taken.
        sample (int):
            Number of samples used in estimation.
        convert_from_log_scale (bool):  
            Whether to take the exp() of the result

    Returns:
        (numpy.ndarray):
            Percentiles of ``poly`` with ``Q.shape=poly.shape+q.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([0.05*q0, 0.2*q1, 0.01*q0*q1])
        >>> chaospy.Perc(poly, [0, 5, 50, 95, 100], dist).round(2)
        array([[ 0.  , -3.29, -5.3 ],
               [ 0.  , -0.64, -0.04],
               [ 0.03, -0.01, -0.  ],
               [ 0.15,  0.66,  0.04],
               [ 1.61,  3.29,  5.3 ]])

    """
    poly = chaospy.aspolynomial(poly)
    shape = poly.shape
    poly = poly.ravel()

    q = numpy.asarray(q).ravel() / 100.0
    dim = len(dist)

    # Interior
    Z = dist.sample(sample, **kws).reshape(len(dist), sample)
    poly1 = poly(*Z)

    # Min/max
    ext = numpy.mgrid[(slice(0, 2, 1),) * dim].reshape(dim, 2 ** dim).T
    ext = numpy.where(ext, dist.lower, dist.upper).T
    poly2 = poly(*ext)
    poly2 = numpy.array([_ for _ in poly2.T if not numpy.any(numpy.isnan(_))]).T

    # Finish
    if poly2.shape:
        poly1 = numpy.concatenate([poly1, poly2], -1)
    if convert_from_log_scale:
        poly1 = numpy.exp(poly1)
    samples = poly1.shape[-1]
    poly1.sort()
    out = poly1.T[numpy.asarray(q * (samples - 1), dtype=int)]
    out = out.reshape(q.shape + shape)

    return out
