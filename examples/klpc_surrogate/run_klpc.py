#!/usr/bin/env python

import os
from matplotlib import pyplot
import numpy
from scipy.special import ndtri

from ensembleperturbation.uncertainty_quantification.ensemble_array import ensemble_array, read_combined_hdf
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import karhunen_loeve_expansion, karhunen_loeve_prediction
from ensembleperturbation.uncertainty_quantification.polynomial_chaos import build_pc_expansion, evaluate_pc_expansion, evaluate_pc_sensitivity

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
neig = 0.90 # gives us 6 modes
#neig = 0.99 # gives us 23 modes

## Evaluating the KL modes
# ymean is the average field                                                 : size (ngrid,)
# kl_modes is the KL modes ('principal directions')                          : size (ngrid,neig)
# eigval is the eigenvalue vector                                            : size (neig,)
# xi are the samples for the KL coefficients                                 : size (nens, neig)
ymean, kl_modes, eigval, xi = karhunen_loeve_expansion(ymodel,neig=neig,plot=False)

# evaluate the fit of the KL prediction
# ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
ypred = karhunen_loeve_prediction(ymean, kl_modes, eigval, xi, ymodel)

# Build PC for each mode in xi (each mode has nens values)
pc_type = 'HG' # Hermite-Gauss chaos
lambda_reg = 0 # regularization lambda
neig = xi.shape[1] # number of eigenvalues
pc_dim = np_input.shape[1] # dimension of the PC expansion
tot_sens_all = numpy.empty((neig,pc_dim))
main_sens_all = numpy.empty((neig,pc_dim))
for k, qoi in enumerate(xi.transpose()):
    numpy.savetxt('qoi.dat', qoi)

    # compare accuracy of 2nd or 3rd order polynomials
    for poly_order in [2,3]: 
        # Builds second order PC expansion for the each mode
        build_pc_expansion(x_filename='xdata.dat',y_filename='qoi.dat',
                           output_filename='coeff' + str(k+1) + '.dat',
                           pc_type=pc_type,poly_order=poly_order,
                           lambda_regularization=lambda_reg)

        # Evaluates the constructed PC at the input for comparison
        evaluate_pc_expansion(x_filename='xdata.dat',output_filename='ydata.dat',
                              parameter_filename='coeff' + str(k+1) + '.dat',
                              pc_type=pc_type,poly_order=poly_order)
        qoi_pc = numpy.loadtxt('ydata.dat')

        # shows comparison of predicted against "real" result
        pyplot.plot(qoi, qoi_pc, 'o',label='poly order = ' +  str(poly_order))
    pyplot.plot([-2,3],[-2,3], 'k--', lw=1)
    pyplot.gca().set_xlabel('predicted')
    pyplot.gca().set_ylabel('actual')
    pyplot.legend()
    pyplot.savefig('mode-' + str(k+1))
    pyplot.close()
        
    # Evaluates the constructed PC at the input for comparison
    evaluate_pc_sensitivity(parameter_filename='coeff' + str(k+1) + '.dat',
                            pc_type=pc_type,pc_dimension=pc_dim,poly_order=poly_order)
    tot_sens_all[k,:] = numpy.loadtxt('totsens.dat')
    main_sens_all[k,:] = numpy.loadtxt('mainsens.dat')
  
for vdx, variable in enumerate(df_input.columns):
    tot_sens = tot_sens_all[:,vdx] 
    main_sens = main_sens_all[:,vdx] 
    print(variable + '-> main sensitivity')
    #print(tot_sens)
    print(main_sens)

# now do something with our qoi_pcs
# WHAT NEEDS TO BE DONE: Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
#poly_order = 3
# update the input matrix
#np_input = np_input
#numpy.savetxt('xdata.dat', np_input) #because pce_eval expects xdata.dat as input
#for k in range(neig):
#
#    # Evaluates the constructed PC at the input for comparison
#    evaluate_pc_expansion(x_filename='xdata.dat',parameter_filename='coeff' + str(k+1) + '.dat',
#                          output_filename='ydata.dat',pc_type=pc_type,poly_order=poly_order)
#    qoi_pc = numpy.loadtxt('ydata.dat')
# 
# evaluate the fit of the KL prediction
# ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
#ypred = karhunen_loeve_prediction(ymean, kl_modes, eigval, xi_pc, ymodel)
