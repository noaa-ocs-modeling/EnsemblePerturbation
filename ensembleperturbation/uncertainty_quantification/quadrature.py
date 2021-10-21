import chaospy
import numpoly
import xarray

from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('quadrature')


def fit_surrogate_to_quadrature(
    samples: xarray.DataArray,
    polynomials: numpoly.PolyLike,
    perturbations: xarray.DataArray,
    weights: xarray.DataArray,
):
    # create surrogate models for selected nodes
    LOGGER.info(f'fitting surrogate to {samples.shape} samples')
    try:
        surrogate_model = chaospy.fit_quadrature(
            orth=polynomials, nodes=perturbations.T, weights=weights, solves=samples,
        )
    except AssertionError:
        if (
            len(perturbations['run']) != len(weights)
            or len(weights) != len(samples)
            or len(samples) != len(perturbations['run'])
        ):
            raise AssertionError(
                f'{len(perturbations["run"])} != {len(weights)} != {len(samples)}'
            )
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
    predicted_percentiles = chaospy.Perc(
        poly=surrogate_model, q=percentiles, dist=distribution, sample=samples.shape[1],
    )

    predicted_percentiles = xarray.DataArray(
        predicted_percentiles,
        coords={
            'quantile': percentiles,
            **{coord: values for coord, values in samples.coords.items() if coord != 'run'},
        },
        dims=('quantile', *(dim for dim in samples.dims if dim != 'run')),
    )

    return predicted_percentiles
