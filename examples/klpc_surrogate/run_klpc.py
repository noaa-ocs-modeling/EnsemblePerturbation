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
    evaluate_pc_distribution_function,
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
neig = xi.shape[1] # number of eigenvalues
pc_dim = np_input.shape[1]  # dimension of the PC expansion
num_samples = 1000 # number of times to sample the PC expansion to get PDF
pdf_bins = 1000    # number of PDF bins
sens_types = ['main', 'total']             # List of sensitivity types to keep
distribution_all = numpy.empty((neig, pdf_bins, 3)) # Storing PDF/CDF of each KL mode
sens_all = numpy.empty((neig, pc_dim, 3))  # Storing Sensitivities of each KL mode
for mode, qoi in enumerate(xi.transpose()):
    numpy.savetxt('qoi.dat', qoi)

    # compare accuracy of 2nd or 3rd order polynomials
    for poly_order in [2, 3]:
        # Builds second order PC expansion for the each mode
        build_pc_expansion(
            x_filename='xdata.dat',
            y_filename='qoi.dat',
            output_filename=f'coeff{mode + 1}.dat',
            pc_type=pc_type,
            poly_order=poly_order,
            lambda_regularization=lambda_reg,
        )

        # Evaluates the constructed PC for the training data for comparison
        qoi_pc = evaluate_pc_expansion(
            x_filename='xdata.dat',
            output_filename='ydata.dat',
            parameter_filename=f'coeff{mode + 1}.dat',
            pc_type=pc_type,
            poly_order=poly_order,
        )

        # shows comparison of predicted against "real" result
        pyplot.plot(qoi, qoi_pc, 'o', label='poly order = ' + str(poly_order))
    pyplot.plot([-2, 3], [-2, 3], 'k--', lw=1)
    pyplot.gca().set_xlabel('predicted')
    pyplot.gca().set_ylabel('actual')
    pyplot.title(f'mode-{mode + 1}')
    pyplot.legend()
    pyplot.savefig(f'mode-{mode + 1}')
    pyplot.close()

    # Evaluates the Sobol sensitivities for the 3rd order PC
    main_sens, joint_sens, total_sens = evaluate_pc_sensitivity(
        parameter_filename=f'coeff{mode + 1}.dat',
        pc_type=pc_type,
        pc_dimension=pc_dim,
        poly_order=poly_order,
    )
    sens_all[mode, :, 0] = main_sens
    sens_all[mode, :, 1] = total_sens

    # Evaluates the PDF/CDF of the constructed 3rd order PC
    xvalue, pdf, cdf = evaluate_pc_distribution_function(
        parameter_filename=f'coeff{mode + 1}.dat',
        pc_type=pc_type,
        pc_dimension=pc_dim,
        poly_order=poly_order,
        num_samples=num_samples,
        pdf_bins=pdf_bins,
    )
    distribution_all[mode, :, 0] = xvalue
    distribution_all[mode, :, 1] = pdf
    distribution_all[mode, :, 2] = cdf

# Plotting the sensitivities
for sdx,sens_label in enumerate(sens_types):
    lineObjects = pyplot.plot(sens_all[:, :, sdx].squeeze())
    pyplot.gca().set_xlabel('mode number')
    pyplot.gca().set_ylabel('Sobol sensitivty')
    pyplot.title(sens_label + '_sensitivity')
    pyplot.legend(lineObjects, dataframes[keys[0]].columns)
    pyplot.savefig(sens_label + '_sensitivity')
    pyplot.close()

# Plotting the PDF/CDFs of each mode
for pdx,df in enumerate(['PDF','CDF']):
    for mode in range(neig):
        pyplot.plot(
            distribution_all[mode, :, 0].squeeze(),
            distribution_all[mode, :, pdx+1].squeeze(),
            label=f'KL Mode-{mode+1}'
        )
    pyplot.gca().set_xlabel('x')
    pyplot.gca().set_ylabel('P')
    pyplot.title(df + ' of PC surrogate for each KL mode')
    pyplot.legend()
    pyplot.grid()
    pyplot.savefig('KLPC_' + df)
    pyplot.close()
