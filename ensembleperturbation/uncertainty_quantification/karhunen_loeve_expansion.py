import numpy
from matplotlib import pyplot

def karhunen_loeve_expansion(ymodel, neig = None, plot: bool = False):

    # get the shape of the data
    ngrid, nens = ymodel.shape
    
    if neig is None:
        neig = ngrid
    elif isinstance(neig,int):
        neig = min(neig,ngrid)
    elif isinstance(neig,float):
        assert neig <= 1.0 and neig >= 0.0, 'specify 0.0 <= neig <= 1.0'

    # evaluate weights and eigen values
    mean_vector = numpy.mean(ymodel, axis=1)
    covariance = numpy.cov(ymodel)

    weights = trapezoidal_rule_weights(length=ngrid)

    eigen_values, eigen_vectors = karhunen_loeve_eigen_values(
        covariance=covariance, weights=weights
    )

    # Karhunen–Loève modes ('principal directions')
    modes = karhunen_loeve_modes(eigen_vectors=eigen_vectors, weights=weights)
    eigen_values[eigen_values < 1.0e-14] = 1.0e-14

    relative_diagonal = karhunen_loeve_relative_diagonal(
        karhunen_loeve_modes=modes, eigen_values=eigen_values, covariance=covariance
    )

    xi = karhunen_loeve_coefficient_samples(
        data=ymodel,
        eigen_values=eigen_values,
        eigen_vectors=eigen_vectors,
    )

    # re-ordering the matrices (mode-1 first) 
    xi = xi[:, ::-1]
    modes = modes[:, ::-1]
    eigen_values = eigen_values[::-1]
  
    # get desired modes 
    if isinstance(neig,float):
        # determine neig that make up neig decimal fraction of the variance explained
        target = neig*sum(eigen_values)
        eig_sum = 0
        for neig in range(ngrid):
            eig_sum = eig_sum + eigen_values[neig]
            if eig_sum >= target:
                break 
        xi = xi[:, :neig]
        eigen_values = eigen_values[:neig]
        modes = modes[:, :neig]
    else:
        # get neig requested modes 
        xi = xi[:, :neig]
        eigen_values = eigen_values[:neig]
        modes = modes[:, :neig]

    # evaluating the model prediction based on neig modes
    # now ypred is ngrid x nens just like ymodel
    ypred = mean_vector + numpy.dot(numpy.dot(xi, numpy.diag(numpy.sqrt(eigen_values))), modes.T)
    ypred = ypred.T

    if plot:
        pyplot.figure(figsize=(12, 9))
        pyplot.plot(range(1, neig+1), eigen_values, 'o-')
        pyplot.gca().set_xlabel('x')
        pyplot.gca().set_ylabel('Eigenvalue')
        pyplot.savefig('eig.png')
        pyplot.gca().set_yscale('log')
        pyplot.savefig('eig_log.png')
        pyplot.close()

        pyplot.figure(figsize=(12, 9))
        pyplot.plot(range(ngrid), mean_vector, label='Mean')
        for imode in range(neig):
            pyplot.plot(range(ngrid), modes[:, imode], label='Mode ' + str(imode + 1))
        pyplot.gca().set_xlabel('x')
        pyplot.gca().set_ylabel('KL Modes')
        pyplot.legend()
        pyplot.savefig('KLmodes.png')
        pyplot.close()

        # Plot to make sure ypred and ymodel are close
        pyplot.plot(ymodel, ypred, 'o')
        pyplot.gca().set_xlabel('actual')
        pyplot.gca().set_ylabel('prediction')
        pyplot.savefig('KLfit.png')
        pyplot.close()

    return ypred, mean_vector, modes, eigen_values, xi

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
