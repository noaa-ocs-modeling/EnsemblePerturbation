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
    pinput, output = ensemble_array(
        *read_combined_hdf(
            filename=r'C:\Data\COASTAL_Act\runs\run_20210812_florence_multivariate_besttrack_250msubset_40members.h5'
        )
    )

    numpy.nan_to_num(output, copy=False)

    ymodel = output[:, ::100].T  # ymodel has a shape of ngrid x nens
    ngrid, nens = ymodel.shape

    # average of fields
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
        data=ymodel, eigen_values=eigen_values, eigen_vectors=eigen_vectors,
    )

    xi = xi[:, :-1]
    modes = modes[:, :-1]
    eigen_values = eigen_values[:-1]

    pyplot.figure(figsize=(12, 9))
    pyplot.plot(range(1, ngrid + 1), eigen_values, 'o-')
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
    pyplot.plot(range(ngrid), mean_vector, label='Mean')
    for imode in range(ngrid):
        pyplot.plot(range(ngrid), modes[:, imode], label='Mode ' + str(imode + 1))
    pyplot.gca().set_xlabel('x')
    pyplot.gca().set_ylabel('KL Modes')
    pyplot.legend()
    pyplot.savefig('KLmodes.png')
    pyplot.close()

    # pick the first neig eigenvalues, look at rel_diag array or eig.png to choose how many eigenmodes you should pick without losing much accuracy
    # can go all the way to neig = ngrid, in which case one should exactly recover ypred = ymodel
    neig = 25
    xi = xi[:, :neig]
    eigen_values = eigen_values[:neig]
    modes = modes[:, :neig]

    # Evaluate KL expansion using the same xi.
    #
    # WHAT NEEDS TO BE DONE: pick each column of xi (neig of them) and build PC surrogate for it like in run_pc.py (or feed the xi matrix to uqpc/uq_pc.py which I think Zach has looked at?), and then replace the xi below with its PC approximation xi_pc. Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
    #
    ypred = mean_vector + numpy.dot(
        numpy.dot(xi, numpy.diag(numpy.sqrt(eigen_values))), modes.T
    )
    ypred = ypred.T
    # now ypred is ngrid x nens just like ymodel

    # Plot to make sure ypred and ymodel are close
    pyplot.plot(ymodel, ypred, 'o')
    pyplot.show()
