from os import PathLike
from pathlib import Path
import pickle
from typing import Union

import geopandas
from matplotlib import pyplot
import numpy as np
from sklearn.decomposition import PCA

from ensembleperturbation.plotting.geometry import plot_points, plot_surface
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('karhunen_loeve')


def karhunen_loeve_expansion(
    ymodel: np.ndarray,
    neig,
    method: Union[float, str] = None,
    output_directory: PathLike = None,
):
    if output_directory is not None and not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    # get the shape of the data
    nens, ngrid = ymodel.shape

    if neig is None:
        neig = ngrid
    elif isinstance(neig, int):
        neig = min(neig, ngrid)
    elif isinstance(neig, float):
        assert neig <= 1.0 and neig >= 0.0, 'specify 0.0 <= neig <= 1.0'

    if method == 'PCA':
        LOGGER.info(f'Using sklearn PCA decomposition method')
        # Using the scikit PCA (same as KL in discrete space)
        pca_obj = PCA(n_components=neig, random_state=666, whiten=True)
        pca_obj.fit(ymodel)
        kl_dict = {
            'mean_vector': pca_obj.mean_,
            'modes': pca_obj.components_,
            'eigenvalues': pca_obj.explained_variance_,
            'neig': len(pca_obj.explained_variance_),
            'samples': pca_obj.transform(ymodel),
        }

    else:
        LOGGER.info(f'Using native KL decomposition method')
        ymodel = ymodel.T

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
        #
        relative_diagonal = karhunen_loeve_relative_diagonal(
            karhunen_loeve_modes=modes, eigen_values=eigen_values, covariance=covariance
        )
        #
        xi = karhunen_loeve_coefficient_samples(
            data=ymodel, eigen_values=eigen_values, eigen_vectors=eigen_vectors,
        )
        #
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
            neig = neig + 1

        # form into KL dictionary
        kl_dict = {
            'mean_vector': mean_vector,
            'modes': modes[:, :neig].T,
            'eigenvalues': eigen_values[:neig],
            'neig': neig,
            'samples': xi[:, :neig],
        }

        # plot the eigenvalues and KL modes, and save to file
    if output_directory is not None:
        filename = output_directory / 'karhunen_loeve.pkl'
        with open(filename, 'wb') as kl_handle:
            LOGGER.info(f'saving Karhunen-Loeve expansion to "{filename}"')
            pickle.dump(kl_dict, kl_handle)

        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

        axis.plot(range(1, kl_dict['neig'] + 1), kl_dict['eigenvalues'], 'o-')

        axis.set_xlabel('x')
        axis.set_ylabel('Eigenvalue')

        figure.savefig(output_directory / 'KL_eigenvalues.png', dpi=200, bbox_inches='tight')
        pyplot.close()

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
                    * kl_dict['modes'][mode_index, z_index]
                )

    return klpc_coefficients  #: size (num_points, num_coefficients)


def karhunen_loeve_prediction(
    kl_dict: dict,
    samples=None,
    actual_values=None,
    ensembles_to_plot=None,
    element_table=None,
    plot_directory: PathLike = None,
):
    """
    Evaluating the KL model prediction (against actual values if provided)
    
    """

    if plot_directory is not None and not isinstance(plot_directory, Path):
        plot_directory = Path(plot_directory)

    if samples is None:
        samples = kl_dict['samples']

    kl_prediction = kl_dict['mean_vector'] + np.dot(
        np.dot(samples, np.diag(np.sqrt(kl_dict['eigenvalues']))), kl_dict['modes']
    )

    if actual_values is not None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

        # Plot to make sure kl_prediction and actual_values are close
        axis.plot(actual_values.values.T, kl_prediction.T, '.', markersize=1)

        axis.set_xlabel('actual')
        axis.set_ylabel('reconstructed')
        axis.title.set_text(f'KL fit for each ensemble')

        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        axis.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        axis.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

        figure.savefig(
            plot_directory / f'KL_fit.png', dpi=200, bbox_inches='tight',
        )
        pyplot.close()

        bounds = np.array(
            [
                actual_values['x'].min(),
                actual_values['y'].min(),
                actual_values['x'].max(),
                actual_values['y'].max(),
            ]
        )
        vmax = np.round_(actual_values.quantile(0.98), decimals=1)
        vmin = min(0.0, np.round_(actual_values.quantile(0.02), decimals=1))
        sources = {'actual': actual_values, 'reconstructed': kl_prediction}
        for example in ensembles_to_plot:
            figure = pyplot.figure()
            figure.set_size_inches(10, 10 / 1.61803398875)
            figure.suptitle(f'KL reconstruction comparison, ensemble #{example}')
            index = 0
            for source, value in sources.items():
                index += 1
                map_axis = figure.add_subplot(2, len(sources), index)
                map_axis.title.set_text(f'{source}')
                countries = geopandas.read_file(
                    geopandas.datasets.get_path('naturalearth_lowres')
                )
                map_axis.set_xlim((bounds[0], bounds[2]))
                map_axis.set_ylim((bounds[1], bounds[3]))
                countries.plot(color='lightgrey', ax=map_axis)

                if element_table is None:
                    im = plot_points(
                        np.vstack(
                            (actual_values['x'], actual_values['y'], value[example, :])
                        ).T,
                        axis=map_axis,
                        add_colorbar=False,
                        vmax=vmax,
                        vmin=vmin,
                        s=1,
                        extend='both',
                    )
                else:
                    im = plot_surface(
                        points=np.vstack(
                            (actual_values['x'], actual_values['y'], value[example, :])
                        ).T,
                        element_table=element_table.values,
                        axis=map_axis,
                        add_colorbar=False,
                        levels=np.linspace(vmin, vmax, 25 + 1),
                        extend='both',
                    )

            pyplot.subplots_adjust(wspace=0.02, right=0.96)
            cax = pyplot.axes([0.95, 0.55, 0.015, 0.3])
            cbar = figure.colorbar(im, extend='both', cax=cax)

            figure.savefig(
                plot_directory / f'KL_ensemble{example}.png', dpi=200, bbox_inches='tight',
            )
            pyplot.close()

    return kl_prediction


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
