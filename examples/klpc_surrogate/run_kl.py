#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt


def KL(data):
# data is ngrid x nens
    mean = np.mean(data, axis=1)
    cov = np.cov(data)
    ngrid = data.shape[0]

    # Set trapesoidal rule weights
    weights = np.ones(ngrid)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights = np.sqrt(weights)

    cov_sc = np.outer(weights, weights) * cov

    eigval, eigvec = np.linalg.eigh(cov_sc)

    kl_modes = eigvec / weights.reshape(-1, 1) # ngrid, neig
    eigval[eigval<1.e-14] = 1.e-14



    tmp = kl_modes[:, ::-1] * np.sqrt(eigval[::-1])
    rel_diag = (np.cumsum(tmp * tmp, axis=1) + 0.0) / (np.diag(cov).reshape(-1, 1) + 0.0)

    xi = np.dot(data.T - mean, eigvec * weights.reshape(-1, 1)) / np.sqrt(eigval) #nens, neig

    xi = xi[:, ::-1]
    kl_modes = kl_modes[:, ::-1]
    eigval = eigval[::-1]
    plt.figure(figsize=(12,9))
    plt.plot(range(1,ngrid+1),eigval, 'o-')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('Eigenvalue')
    plt.savefig('eig.png')
    plt.gca().set_yscale('log')
    plt.savefig('eig_log.png')

    plt.close()

    # fig = figure(figsize=(12,9))
    # plot(range(1,neig+1),eigValues, 'o-')
    # yscale('log')
    # xlabel('x')
    # ylabel('Eigenvalue')
    # savefig('eig_log.png')
    # clf()

    plt.figure(figsize=(12,9))
    plt.plot(range(ngrid),mean, label='Mean')
    for imode in range(ngrid):
        plt.plot(range(ngrid),kl_modes[:,imode], label='Mode '+str(imode+1))
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('KL Modes')
    plt.legend()
    plt.savefig('KLmodes.png')
    plt.close()

    return mean, kl_modes, eigval, xi, rel_diag, weights

##############################################################################
##############################################################################
##############################################################################

output = np.loadtxt('output.txt')

output = np.nan_to_num(output)


ymodel = output[:, ::100].T # ymodel has a shape of ngrid x nens
ngrid, nens = ymodel.shape

# mean is the average field, size (ngrid,)
# kl_modes is the KL modes ('principal directions') of size (ngrid, ngrid)
# eigval is the eigenvalue vector, size (ngrid,)
# xi are the samples for the KL coefficients, size (nens, ngrid)
mean, kl_modes, eigval, xi, rel_diag, weights = KL(ymodel)


# pick the first neig eigenvalues, look at rel_diag array or eig.png to choose how many eigenmodes you should pick without losing much accuracy
# can go all the way to neig = ngrid, in which case one should exactly recover ypred = ymodel
neig = 25
xi = xi [:, :neig]
eigval = eigval[:neig]
kl_modes = kl_modes[:, :neig]

# Evaluate KL expansion using the same xi.
#
# WHAT NEEDS TO BE DONE: pick each column of xi (neig of them) and build PC surrogate for it like in run_pc.py (or feed the xi matrix to uqpc/uq_pc.py which I think Zach has looked at?), and then replace the xi below with its PC approximation xi_pc. Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
#
ypred = mean + np.dot(np.dot(xi, np.diag(np.sqrt(eigval))), kl_modes.T)
ypred = ypred.T
# now ypred is ngrid x nens just like ymodel

# Plot to make sure ypred and ymodel are close
plt.plot(ymodel, ypred, 'o')
plt.show()
