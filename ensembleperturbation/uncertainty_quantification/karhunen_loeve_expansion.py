import numpy


def trapezoidal_rule_weights(length: int):
    # Set trapesoidal rule weights
    weights = numpy.full(length, fill_value=1)
    weights[[0, -1]] = 0.5
    return numpy.sqrt(weights)


def karhunen_loeve_eigen_values(
    covariance: numpy.ndarray, weights: numpy.ndarray
) -> (numpy.ndarray, numpy.ndarray):
    return numpy.linalg.eigh(numpy.outer(weights, weights) * covariance)


def karhunen_loeve_modes(eigen_vectors: numpy.ndarray, weights: numpy.ndarray):
    return eigen_vectors / weights.reshape(-1, 1)  # ngrid, neig


def karhunen_loeve_relative_diagonal(
    karhunen_loeve_modes: numpy.ndarray, eigen_values: numpy.ndarray, covariance: numpy.ndarray
):
    cumulative_sum = (
        numpy.cumsum(
            (karhunen_loeve_modes[:, :-1] * numpy.sqrt(eigen_values[:-1])) ** 2, axis=1
        )
        + 0.0
    )
    diagonal = numpy.diag(covariance).reshape(-1, 1) + 0.0
    return cumulative_sum / diagonal


def karhunen_loeve_coefficient_samples(
    data: numpy.ndarray, eigen_values: numpy.ndarray, eigen_vectors: numpy.ndarray,
):
    """
    get samples for the Karhunen–Loève coefficients from the given data

    :param data: array of data of shape `ngrid x nens`
    :param eigen_values: 1D array of eigen values
    :param eigen_vectors: 2D array of eigen vectors
    :return: samples for the Karhunen–Loève coefficients
    """
    # nens, neig
    return numpy.dot(
        data.T - numpy.mean(data, axis=1),
        eigen_vectors * trapezoidal_rule_weights(data.shape[0]).reshape(-1, 1),
    ) / numpy.sqrt(eigen_values)
