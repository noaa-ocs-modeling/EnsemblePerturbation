from typing import Union

from matplotlib import pyplot
import numpy as np


def karhunen_loeve_expansion(ymodel, neig: Union[int, float] = None, plot: bool = False):
    # get the shape of the data
    ngrid, nens = ymodel.shape

    if neig is None:
        neig = ngrid
    elif isinstance(neig, int):
        neig = min(neig, ngrid)
    elif isinstance(neig, float):
        assert neig <= 1.0 and neig >= 0.0, 'specify 0.0 <= neig <= 1.0'

    # evaluate weights and eigen values
    mean_vector = np.mean(ymodel, axis=1)
    covariance = np.cov(ymodel)

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
        data=ymodel, eigen_values=eigen_values, eigen_vectors=eigen_vectors,
    )

    # re-ordering the matrices (mode-1 first)
    xi = xi[:, ::-1]
    modes = modes[:, ::-1]
    eigen_values = eigen_values[::-1]

    # get desired modes
    if isinstance(neig, float):
        # determine neig that make up neig decimal fraction of the variance explained
        target = neig * sum(eigen_values)
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

    if plot:
        pyplot.figure()
        pyplot.plot(range(1, neig + 1), eigen_values, 'o-')
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

    # return KL dictionary
    kl_dict = {
        'mean_vector': mean_vector,
        'modes': modes,
        'eigenvalues': eigen_values,
        'samples': xi,
    }
    return kl_dict


def karhunen_loeve_pc_coefficients(
    kl_dict: dict, pc_coefficients: np.ndarray,
):
    """
    Get the joint karhunen_loeve Polynomial chaos polynomial coefficients
    from Eq (6) in Sargsyan, K. and Ricciuto D. (2021):
    sum_{k=0}^{K} (kroneckerdelta_{k0}*mean(f(z)) +  
                   sum_{j=0}^{L} [c_{jk}*\sqrt(ev_j)*ef_j(z)]) 
    K -> num PC coefficients
    L -> num KL modes
    c -> PC coefficient 
    ev -> eigenvalue
    ef -> eigenfunction
    """

    # get the coefficients of the PC for each point in z (spatiotemporal dimension)
    num_points = len(kl_dict['mean_vector'])
    num_modes = len(kl_dict['eigenvalues'])
    assert (
        num_modes == pc_coefficients.shape[0]
    ), 'number of kl_dict eigenvalues needs to be equal to the length of the first dimension of pc_coefficients'
    num_coefficients = pc_coefficients.shape[1]
    klpc_coefficients = np.zeros((num_points, num_coefficients))
    klpc_coefficients[:, 0] = kl_dict['mean_vector']
    for z_index in range(num_points):
        for coef_index in range(num_coefficients):
            for mode_index in range(num_modes):
                klpc_coefficients[z_index, coef_index] += (
                    pc_coefficients[mode_index, coef_index]
                    * np.sqrt(kl_dict['eigenvalues'][mode_index])
                    * kl_dict['modes'][z_index, mode_index]
                )

    return klpc_coefficients  #: size (num_points, num_coefficients)


def karhunen_loeve_prediction(kl_dict: dict, samples=None, ymodel=None):
    """
    Evaluating the model prediction based on KL modes
    
    """

    if samples is None:
        samples = kl_dict['samples']

    ypred = kl_dict['mean_vector'] + np.dot(
        np.dot(samples, np.diag(np.sqrt(kl_dict['eigenvalues']))), kl_dict['modes'].T
    )
    ypred = ypred.T

    if ymodel is not None:
        # Plot to make sure ypred and ymodel are close
        pyplot.plot(ymodel, ypred, 'o')
        pyplot.gca().set_xlabel('actual')
        pyplot.gca().set_ylabel('prediction')
        pyplot.savefig('KLfit.png')
        pyplot.close()

    return ypred


def trapezoidal_rule_weights(length: int):
    # Set trapezoidal rule weights
    weights = np.ones(length)
    weights[[0, -1]] = 0.5
    return np.sqrt(weights)


def karhunen_loeve_eigen_values(
    covariance: np.ndarray, weights: np.ndarray
) -> (np.ndarray, np.ndarray):
    return np.linalg.eigh(np.outer(weights, weights) * covariance)


def karhunen_loeve_modes(eigen_vectors: np.ndarray, weights: np.ndarray):
    return eigen_vectors / weights.reshape(-1, 1)  # ngrid, neig


def karhunen_loeve_relative_diagonal(
    karhunen_loeve_modes: np.ndarray, eigen_values: np.ndarray, covariance: np.ndarray
):
    cumulative_sum = (
        np.cumsum((karhunen_loeve_modes[:, :-1] * np.sqrt(eigen_values[:-1])) ** 2, axis=1)
        + 0.0
    )
    diagonal = np.diag(covariance).reshape(-1, 1) + 0.0
    return cumulative_sum / diagonal


def karhunen_loeve_coefficient_samples(
    data: np.ndarray, eigen_values: np.ndarray, eigen_vectors: np.ndarray,
):
    """
    get samples for the Karhunen–Loève coefficients from the given data

    :param data: array of data of shape `ngrid x nens`
    :param eigen_values: 1D array of eigen values
    :param eigen_vectors: 2D array of eigen vectors
    :return: samples for the Karhunen–Loève coefficients
    """
    # nens, neig
    return np.dot(
        data.T - np.mean(data, axis=1),
        eigen_vectors * trapezoidal_rule_weights(data.shape[0]).reshape(-1, 1),
    ) / np.sqrt(eigen_values)
