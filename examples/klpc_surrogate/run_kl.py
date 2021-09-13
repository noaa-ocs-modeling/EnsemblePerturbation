"""
run_kl.py : an example of Karhunen-Loeve construction.
I found a pure python implementation (it is bunch of linear algebra), so this does not rely on UQTk.
This is a simple demo as a proof of concept - see the comments I left.
Basically, instead of the full 200K output points, you should build PC surrogate for the first, say neig xi's (KL coefficients corresponding to first neig eigenmodes).
As I said, this is a glorified principal components analysis (PCA) - feel free to use your own PCA, but the idea is the same.
Now, the caveat: in run_kl.py, I picked every 100th grid point, otherwise the code chokes (KL relies on eigendecomposition that is impossible for 200K grid points, since it requires 200K by 200K matrix inversion). There are sparse covariance methods at the expense of more loss of accuracy, and they are quite complex.
The easiest way-around for me was to pick every, say, 100th grid point.
You can check out the map with read_hdf.py or your notebook with every 100th point and see if it captures the interesting locations...
Otherwise, do your own downselection (sparsification) of the regional grid looking at the longitude/latitude pairs more meaningfully.
- Khachik
"""

from matplotlib import pyplot
import numpy

from ensembleperturbation.uncertainty_quantification.ensemble_array import (
    ensemble_array,
    read_combined_hdf,
)
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import (
    karhunen_loeve_coefficient_samples,
    karhunen_loeve_eigen_values,
    karhunen_loeve_modes,
    karhunen_loeve_relative_diagonal,
    trapezoidal_rule_weights,
)

if __name__ == '__main__':
    plot_eigen_values = False
    plot_surrogate = True

    pinput, output = ensemble_array(
        *read_combined_hdf(
            filename=r'run_20210812_florence_multivariate_besttrack_250msubset_40members.h5'
        )
    )

    numpy.nan_to_num(output, copy=False)

    model_results = output[:, ::100].T  # ymodel has a shape of ngrid x nens
    num_nodes, num_ensemble_members = model_results.shape

    # average of fields
    mean_vector = numpy.mean(model_results, axis=1)
    covariance = numpy.cov(model_results)

    node_weights = trapezoidal_rule_weights(length=num_nodes)

    eigen_values, eigen_vectors = karhunen_loeve_eigen_values(
        covariance=covariance, weights=node_weights
    )

    # Karhunen–Loève modes ('principal directions')
    modes = karhunen_loeve_modes(eigen_vectors=eigen_vectors, weights=node_weights)
    eigen_values[eigen_values < 1.0e-14] = 1.0e-14

    relative_diagonal = karhunen_loeve_relative_diagonal(
        karhunen_loeve_modes=modes, eigen_values=eigen_values, covariance=covariance
    )

    coefficient_samples = karhunen_loeve_coefficient_samples(
        data=model_results, eigen_values=eigen_values, eigen_vectors=eigen_vectors,
    )

    coefficient_samples = coefficient_samples[:, ::-1]
    modes = modes[:, ::-1]
    eigen_values = eigen_values[::-1]

    if plot_eigen_values:
        pyplot.figure(figsize=(12, 9))
        pyplot.plot(range(1, num_nodes + 1), eigen_values, 'o-')
        pyplot.gca().set_xlabel('x')
        pyplot.gca().set_ylabel('Eigenvalue')
        pyplot.savefig('eig.png')
        pyplot.gca().set_yscale('log')
        pyplot.savefig('eig_log.png')

        pyplot.close()

        # fig = figure(figsize=(12,9))
        # plot(range(1,neig+1),eigValues, 'o-')
        # yscale('log')
        # xlabel('x')
        # ylabel('Eigenvalue')
        # savefig('eig_log.png')
        # clf()

        pyplot.figure(figsize=(12, 9))
        pyplot.plot(range(num_nodes), mean_vector, label='Mean')
        for imode in range(num_nodes):
            pyplot.plot(range(num_nodes), modes[:, imode], label='Mode ' + str(imode + 1))
        pyplot.gca().set_xlabel('x')
        pyplot.gca().set_ylabel('KL Modes')
        pyplot.legend()
        pyplot.savefig('KLmodes.png')
        pyplot.close()

    # pick the first neig eigenvalues, look at rel_diag array or eig.png to choose how many eigenmodes you should pick without losing much accuracy
    # can go all the way to neig = ngrid, in which case one should exactly recover ypred = ymodel
    num_eigen_values = 25
    coefficient_samples = coefficient_samples[:, :num_eigen_values]
    eigen_values = eigen_values[:num_eigen_values]
    modes = modes[:, :num_eigen_values]

    # Evaluate KL expansion using the same xi.
    #
    # WHAT NEEDS TO BE DONE: pick each column of xi (neig of them) and build PC surrogate for it like in run_pc.py (or feed the xi matrix to uqpc/uq_pc.py which I think Zach has looked at?), and then replace the xi below with its PC approximation xi_pc.
    # Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration.
    # Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
    #
    predicted_results = mean_vector + numpy.dot(
        numpy.dot(coefficient_samples, numpy.diag(numpy.sqrt(eigen_values))), modes.T
    )
    predicted_results = predicted_results.T
    # now ypred is ngrid x nens just like ymodel

    if plot_surrogate:
        colormap = pyplot.get_cmap('gist_rainbow')
        # Plot to make sure ypred and ymodel are close
        for index in range(len(model_results)):
            pyplot.plot(model_results[index], predicted_results[index], 'o', c=colormap(index / len(model_results)))
        pyplot.plot([0, 6], [0, 6], '--k', label='1:1')
        pyplot.legend()
        pyplot.show()
