import chaospy
import numpoly
from sklearn.linear_model import OrthogonalMatchingPursuit
import xarray

from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('quadrature')


def fit_surrogate(
    samples: xarray.DataArray,
    perturbations: xarray.DataArray,
    polynomials: numpoly.PolyLike,
    quadrature: bool = False,
    quadrature_weights: xarray.DataArray = None,
):
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


def get_percentiles(
    samples: xarray.DataArray,
    percentiles: [float],
    surrogate_model: numpoly.PolyLike,
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
            **{coord: values for coord, values in samples.coords.items() if coord != 'run'},
        },
        dims=('quantile', *(dim for dim in samples.dims if dim != 'run')),
    )

    return surrogate_percentiles
