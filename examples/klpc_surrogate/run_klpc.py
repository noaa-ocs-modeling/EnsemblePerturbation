#!/usr/bin/env python

import os
from matplotlib import pyplot
import numpy
from scipy.special import ndtri

from ensembleperturbation.uncertainty_quantification.ensemble_array import ensemble_array, read_combined_hdf
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import karhunen_loeve_expansion

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
neig = 25

## Evaluating the KL modes
# mean is the average field, size (ngrid,)
# kl_modes is the KL modes ('principal directions') of size (ngrid, ngrid)
# eigval is the eigenvalue vector, size (ngrid,)
# xi are the samples for the KL coefficients, size (nens, ngrid)
ymean, kl_modes, eigval, xi, rel_diag, weights = karhunen_loeve_expansion(ymodel,neig=25,plot=True)
# pick the first neig eigenvalues, look at rel_diag array or eig.png to choose how many eigenmodes you should pick without losing much accuracy

# Evaluate KL expansion using the same xi.
#
# WHAT NEEDS TO BE DONE: pick each column of xi (neig of them) and build PC surrogate for it like in run_pc.py (or feed the xi matrix to uqpc/uq_pc.py which I think Zach has looked at?), and then replace the xi below with its PC approximation xi_pc. Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
#
ypred = ymean + numpy.dot(numpy.dot(xi, numpy.diag(numpy.sqrt(eigval))), kl_modes.T)
ypred = ypred.T
# now ypred is ngrid x nens just like ymodel

# Plot to make sure ypred and ymodel are close
pyplot.plot(ymodel, ypred, 'o')
pyplot.show()

# Pick a QoI of interest, for example, the mean of the whole region
qoi = numpy.mean(np_output, axis=1)

numpy.savetxt('qoi.dat', qoi)
# Builds second order PC expansion for the QoI
#uqtk_cmd = 'regression -x xdata.dat -y qoi.dat -s HG -o 2 -l 0'
#os.system(uqtk_cmd)

# Evaluates the constructed PC at the input for comparison
#uqtk_cmd = 'pce_eval -f coeff.dat -s HG -o 2'
#os.system(uqtk_cmd)
#qoi_pc = numpy.loadtxt('ydata.dat')

# shows comparison of predicted against "real" result
#pyplot.plot(qoi, qoi_pc, 'o')
#pyplot.plot([0,1],[0,1], 'k--', lw=1)
#pyplot.show()
