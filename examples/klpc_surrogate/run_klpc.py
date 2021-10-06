#!/usr/bin/env python

from matplotlib import pyplot
import numpy

from ensembleperturbation.plotting import plot_points
from ensembleperturbation.uncertainty_quantification.ensemble_array import (
    ensemble_array,
    read_combined_hdf,
)
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import (
    karhunen_loeve_expansion,
    karhunen_loeve_prediction,
)
from ensembleperturbation.uncertainty_quantification.polynomial_chaos import (
    build_pc_expansion,
    evaluate_pc_expansion,
    evaluate_pc_sensitivity,
)

##############################################################################
# MAIN SCRIPT ################################################################
##############################################################################

h5name = '../data/florence_40member.h5'
dataframes = read_combined_hdf(h5name)
keys = list(dataframes.keys())

# Load the input/outputs
np_input, np_output = ensemble_array(
    input_dataframe=dataframes[keys[0]], output_dataframe=dataframes[keys[1]],
)
numpy.savetxt('xdata.dat', np_input)  # because pce_eval expects xdata.dat as input

# adjusting the output to have less nodes for now (every 100 points)
points = dataframes[keys[1]][['x', 'y']].to_numpy()
# numpy.nan_to_num(np_output, copy=False)
np_output_subset = np_output[:, ::25]
points_subset = points[::25, :]

# spacing_deg=0.005
# neighbors=50
# get every points spaced every spacing_deg degrees
# mask = sample_points_with_equal_spacing(output_dataframe=df_output, spacing=spacing_deg)
# mask = sample_points_with_equal_spacing(output_dataframe=df_output, neighbors=neighbors)
# np_output_subset = np_output[:, mask]
# remove areas where any of the ensembles was a NaN
mask = numpy.isnan(np_output_subset)
mask = mask.any(axis=0)
ymodel = np_output_subset[:, ~mask].T
points_subset = points_subset[~mask, :]

print('shape of ymodel')
print(ymodel.shape)

# choose neig decimal percent of variance explained to keep
# neig = 0.90 # gives us 4 modes
neig = 0.95  # gives  us 6 modes

## Evaluating the KL modes
# ymean is the average field                                                 : size (ngrid,)
# kl_modes is the KL modes ('principal directions')                          : size (ngrid,neig)
# eigval is the eigenvalue vector                                            : size (neig,)
# xi are the samples for the KL coefficients                                 : size (nens, neig)
ymean, kl_modes, eigval, xi = karhunen_loeve_expansion(ymodel, neig=neig, plot=False)

# evaluate the fit of the KL prediction
# ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
ypred = karhunen_loeve_prediction(ymean, kl_modes, eigval, xi, ymodel)

# plot scatter points to compare ymodel and ypred spatially
for example in range(0, ymodel.shape[1], 5):
    # plot_coastline()
    plot_points(
        numpy.hstack((points_subset, ymodel[:, [example]])),
        save_filename='modeled_zmax' + str(example),
        vmax=3.0,
    )
    pyplot.close()

    # plot_coastline()
    plot_points(
        numpy.hstack((points_subset, ypred[:, [example]])),
        save_filename='predicted_zmax' + str(example),
        vmax=3.0,
    )
    pyplot.close()

# Build PC for each mode in xi (each mode has nens values)
pc_type = 'HG'  # Hermite-Gauss chaos
lambda_reg = 0  # regularization lambda
neig = xi.shape[1]  # number of eigenvalues
pc_dim = np_input.shape[1]  # dimension of the PC expansion
sens_all = numpy.empty((neig, pc_dim, 2))
for k, qoi in enumerate(xi.transpose()):
    numpy.savetxt('qoi.dat', qoi)

    # compare accuracy of 2nd or 3rd order polynomials
    for poly_order in [2, 3]:
        # Builds second order PC expansion for the each mode
        build_pc_expansion(
            x_filename='xdata.dat',
            y_filename='qoi.dat',
            output_filename='coeff' + str(k + 1) + '.dat',
            pc_type=pc_type,
            poly_order=poly_order,
            lambda_regularization=lambda_reg,
        )

        # Evaluates the constructed PC at the input for comparison
        evaluate_pc_expansion(
            x_filename='xdata.dat',
            output_filename='ydata.dat',
            parameter_filename='coeff' + str(k + 1) + '.dat',
            pc_type=pc_type,
            poly_order=poly_order,
        )
        qoi_pc = numpy.loadtxt('ydata.dat')

        # shows comparison of predicted against "real" result
        pyplot.plot(qoi, qoi_pc, 'o', label='poly order = ' + str(poly_order))
    pyplot.plot([-2, 3], [-2, 3], 'k--', lw=1)
    pyplot.gca().set_xlabel('predicted')
    pyplot.gca().set_ylabel('actual')
    pyplot.title('mode-' + str(k + 1))
    pyplot.legend()
    pyplot.savefig('mode-' + str(k + 1))
    pyplot.close()

    # Evaluates the constructed PC at the input for comparison
    evaluate_pc_sensitivity(
        parameter_filename='coeff' + str(k + 1) + '.dat',
        pc_type=pc_type,
        pc_dimension=pc_dim,
        poly_order=poly_order,
    )
    sens_all[k, :, 0] = numpy.loadtxt('mainsens.dat')
    sens_all[k, :, 1] = numpy.loadtxt('totsens.dat')

# print('eigen values = ' + str(eigval))
# for vdx, variable in enumerate(dataframes[keys[0]].columns):
#    tot_sens = tot_sens_all[:,vdx]
#    main_sens = main_sens_all[:,vdx]
#    print(variable + '-> main sensitivity')
#    print(tot_sens)
#    print(main_sens)

sens_labels = ['main', 'total']
for idx in [0, 1]:
    lineObjects = pyplot.plot(sens_all[:, :, idx].squeeze())
    pyplot.gca().set_xlabel('mode number')
    pyplot.gca().set_ylabel('Sobol sensitivty')
    pyplot.title(sens_labels[idx] + '_sensitivity')
    pyplot.legend(lineObjects, dataframes[keys[0]].columns)
    pyplot.savefig(sens_labels[idx] + '_sensitivity')
    pyplot.close()

# now do something with our qoi_pcs
# WHAT NEEDS TO BE DONE: Depends on your final goals, but the surrogate xi_pc and the associated ypred can be evaluated a lot more than 40 times and can be used for sensitivity analysis, moment extraction and model calibration. Essentially you will have a KL+PC spatiotemporal surrogate approximation of your model.
# poly_order = 3
# update the input matrix
# np_input = np_input
# numpy.savetxt('xdata.dat', np_input) #because pce_eval expects xdata.dat as input
# for k in range(neig):
#
#    # Evaluates the constructed PC at the input for comparison
#    evaluate_pc_expansion(x_filename='xdata.dat',parameter_filename='coeff' + str(k+1) + '.dat',
#                          output_filename='ydata.dat',pc_type=pc_type,poly_order=poly_order)
#    qoi_pc = numpy.loadtxt('ydata.dat')
#
# evaluate the fit of the KL prediction
# ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
# ypred = karhunen_loeve_prediction(ymean, kl_modes, eigval, xi_pc, ymodel)
