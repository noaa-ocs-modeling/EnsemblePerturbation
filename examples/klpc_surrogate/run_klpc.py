#!/usr/bin/env python

import os
from matplotlib import pyplot
import numpy
from scipy.special import ndtri

from ensembleperturbation.uncertainty_quantification.ensemble_array import ensemble_array, read_combined_hdf
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import karhunen_loeve_expansion
from ensembleperturbation.uncertainty_quantification.polynomial_chaos import build_pc_expansion, evaluate_pc_expansion

##############################################################################
# MAIN SCRIPT ################################################################
##############################################################################

h5name = '../data/florence_40member.h5'
df_input, df_output = read_combined_hdf(h5name)

# Load the input/outputs
np_input, np_output = ensemble_array(input_dataframe=df_input,output_dataframe=df_output)

# Output this to text to be used in UQtk function
numpy.savetxt('xdata.dat', np_input) #because pce_eval expects xdata.dat as input

# adjusting the output to have less nodes for now (every 100 points)
numpy.nan_to_num(np_output, copy=False)
ymodel = np_output[:, ::100].T # ymodel has a shape of ngrid x nens
# choose neig decimal percent of variance explained to keep
neig = 0.99

## Evaluating the KL modes
# ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
# ymean is the average field                                                 : size (ngrid,)
# kl_modes is the KL modes ('principal directions')                          : size (ngrid,neig)
# eigval is the eigenvalue vector                                            : size (neig,)
# xi are the samples for the KL coefficients                                 : size (nens, neig)
ypred, ymean, kl_modes, eigval, xi = karhunen_loeve_expansion(ymodel,neig=neig,plot=False)
# pick the first neig eigenvalues, look at rel_diag array or eig.png to choose how many eigenmodes you should pick without losing much accuracy

# Evaluate KL expansion using the same xi.
#
# WHAT NEEDS TO BE DONE: pick each column of xi (neig of them) and build PC surrogate for it like in run_pc.py (or feed the xi matrix to uqpc/uq_pc.py which I think Zach has looked at?), and then replace the xi below with its PC approximation xi_pc. Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.

# Build PC for each mode in xi (each mode has nens values)
pc_type = 'HG'  # Hermite-Gauss chaos
poly_order = 2 # polynomial order
lambda_reg = 0 # regularization lambda
for k,qoi in enumerate(xi.transpose()):
    numpy.savetxt('qoi.dat', qoi)
 
    # Builds second order PC expansion for the each mode
    build_pc_expansion(x_filename='xdata.dat',y_filename='qoi.dat',output_filename='coeff.dat',
                       pc_type=pc_type,poly_order=poly_order,lambda_regularization=lambda_reg)

    # Evaluates the constructed PC at the input for comparison
    evaluate_pc_expansion(parameter_filename='coeff.dat',output_filename='ydata.dat',
                          pc_type=pc_type,poly_order=poly_order)
    #qoi_pc = numpy.loadtxt('ydata.dat')

    # shows comparison of predicted against "real" result
    #pyplot.plot(qoi, qoi_pc, 'o')
    #pyplot.plot([0,1],[0,1], 'k--', lw=1)
    #pyplot.show()
