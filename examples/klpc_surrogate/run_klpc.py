#!/usr/bin/env python

from matplotlib import pyplot
import numpy as np

from ensembleperturbation.plotting import plot_points
from ensembleperturbation.uncertainty_quantification.ensemble_array import (
    ensemble_array,
    read_combined_hdf,
)
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import (
    karhunen_loeve_expansion,
    karhunen_loeve_prediction,
    karhunen_loeve_pc_coefficients,
)
from ensembleperturbation.uncertainty_quantification.polynomial_chaos import (
    build_pc_expansion,
    evaluate_pc_distribution_function,
    evaluate_pc_expansion,
    evaluate_pc_multiindex,
    evaluate_pc_sensitivity,
    evaluate_pc_exceedance_probabilities,
    evaluate_pc_exceedance_heights,
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
np.savetxt('xdata.dat', np_input)  # because pce_eval expects xdata.dat as input

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
mask = np.isnan(np_output_subset)
mask = mask.any(axis=0)
ymodel = np_output_subset[:, ~mask].T
points_subset = points_subset[~mask, :]

print('shape of ymodel')
print(ymodel.shape)
ngrid = ymodel.shape[0]
nens = ymodel.shape[1]

# choose neig decimal percent of variance explained to keep
# neig = 0.90 # gives us 4 modes
neig = 0.95  # gives  us 6 modes

## Evaluating the KL mode
# Components of the dictionary:
# mean_vector is the average field                                        : size (ngrid,)
# modes is the KL modes ('principal directions')                          : size (ngrid,neig)
# eigenvalues is the eigenvalue vector                                    : size (neig,)
# samples are the samples for the KL coefficients                         : size (nens, neig)
kl_dict = karhunen_loeve_expansion(ymodel, neig=neig, plot=True)

# evaluate the fit of the KL prediction
# ypred is the predicted value of ymodel -> equal in the limit neig = ngrid  : size (ngrid,nens)
ypred = karhunen_loeve_prediction(kl_dict)

# plot scatter points to compare ymodel and ypred spatially
for example in range(0, nens, 5):
    # plot_coastline()
    plot_points(
        np.hstack((points_subset, ymodel[:, [example]])),
        save_filename='modeled_zmax' + str(example),
        title='modeled zmax, ensemble #' + str(example),
        vmax=3.0,
    )

    # plot_coastline()
    plot_points(
        np.hstack((points_subset, ypred[:, [example]])),
        save_filename='predicted_zmax' + str(example),
        title='predicted zmax, ensemble #' + str(example),
        vmax=3.0,
    )

# Set parameters for the PC
pc_type = 'HG'                # Hermite-Gauss chaos
lambda_reg = 0                # regularization lambda
pc_dim = np_input.shape[1]    # dimension of the PC expansion
pc_order = 3                  # order of the polynomial
mi_type = 'TO'                # PC multi-index type
mi_filename = 'mindex.dat'    # find the multi-index file for later use
evaluate_pc_multiindex(
        multiindex_filename=mi_filename,
        multiindex_type=mi_type,
        pc_dimension=pc_dim,
        poly_order=pc_order,
)

# Build PC for each KL mode (each mode has nens values)
neig = len(kl_dict['eigenvalues']) # number of eigenvalues
for mode, qoi in enumerate(kl_dict['samples'].transpose()):
    np.savetxt('qoi.dat', qoi)

    # Builds second order PC expansion for the each mode
    poly_coefs = build_pc_expansion(
        x_filename='xdata.dat',
        y_filename='qoi.dat',
        output_filename=f'coeff{mode + 1}.dat',
        pc_type=pc_type,
        poly_order=pc_order,
        lambda_regularization=lambda_reg,
    )

    # Enter into array for storing PC coefficients for each KL mode
    if mode == 0:
        ncoefs = len(poly_coefs)                  # number of polynomial coefficients
        pc_coefficients = np.empty((neig,ncoefs)) # array for storing PC coefficients for each KL mode
    pc_coefficients[mode,:] = poly_coefs

    # Evaluates the constructed PC for the training data for comparison
    qoi_pc = evaluate_pc_expansion(
        x_filename='xdata.dat',
        output_filename='ydata.dat',
        parameter_filename=f'coeff{mode + 1}.dat',
        pc_type=pc_type,
        poly_order=pc_order,
    )

    # shows comparison of predicted against "real" result
    pyplot.plot(qoi, qoi_pc, 'o', label='poly order = ' + str(pc_order))
    pyplot.plot([-2, 3], [-2, 3], 'k--', lw=1)
    pyplot.gca().set_xlabel('predicted')
    pyplot.gca().set_ylabel('actual')
    pyplot.title(f'mode-{mode + 1}')
    pyplot.legend()
    pyplot.savefig(f'mode-{mode + 1}')
    pyplot.close()

# Now get the joint KLPC coefficients
klpc_coefficients = karhunen_loeve_pc_coefficients(
    kl_dict=kl_dict, 
    pc_coefficients=pc_coefficients,
)
    
# Get sensitivities and distribution functions
# at each point of the KLP surrgogate

# set some parameters
num_samples = 1000  # number of times to sample the PC expansion to get PDF
pdf_bins = 100      # number of PDF bins
percentiles = np.array([10, 50, 90])  # List of percentiles to extract
klpc_sensitivities = [None] * ngrid
klpc_distributions = [None] * ngrid
for z_index in range(ngrid):
    # save coefficients for this point into parameter file
    np.savetxt('coeff.dat', klpc_coefficients[z_index,:])  

    # evaluate the Sobol sensitivities for the KLPC surrogate
    klpc_sensitivities[z_index] = evaluate_pc_sensitivity(
        parameter_filename='coeff.dat',
        multiindex_filename=mi_filename,
        pc_type=pc_type,
    )

    # evaluate the PDF/CDF of the KLPC surrogate
    klpc_distributions[z_index] = evaluate_pc_distribution_function(
        parameter_filename='coeff.dat',
        multiindex_filename=mi_filename,
        num_samples=num_samples,
        pdf_bins=pdf_bins,
        pc_type=pc_type,
        pc_dimension=pc_dim,
        poly_order=pc_order,
    )

breakpoint()

# Plotting the sensitivities
for sdx, sens_label in enumerate(sens_types):
    lineObjects = pyplot.plot(sens_all[:, :, sdx].squeeze())
    pyplot.gca().set_xlabel('mode number')
    pyplot.gca().set_ylabel('Sobol sensitivty')
    pyplot.title(sens_label + '_sensitivity')
    pyplot.legend(lineObjects, dataframes[keys[0]].columns)
    pyplot.savefig(sens_label + '_sensitivity')
    pyplot.close()

# Plotting the PDF/CDFs of each mode
pc_keys = [*pc_distributions[0]]
pc_keys.remove('x')
for pc_key in pc_keys:
    for mode in range(neig):
        pyplot.plot(
            pc_distributions[mode]['x'],
            pc_distributions[mode][pc_key],
            label=f'KL Mode-{mode+1}',
        )
    pyplot.gca().set_xlabel('x')
    pyplot.gca().set_ylabel('P')
    pyplot.title(pc_key + ' of PC surrogate for each KL mode')
    pyplot.legend()
    pyplot.grid()
    pyplot.savefig('KLPC_' + pc_key)
    pyplot.close()

# Evaluate the percentile of the full KLPC surrogate of zeta_max
zeta_max_percentiles = karhunen_loeve_percentiles(
    percentiles=percentiles, kl_dict=kl_dict, pc_dicts=pc_distributions,
)

# plot scatter points to show zeta_max percentiles
for pdx, perc in enumerate(percentiles):
    plot_points(
        np.hstack((points_subset, zeta_max_percentiles[:, [pdx]])),
        save_filename='zmax_' + str(perc) + '_percentile',
        title=str(perc) + ' percentile maximum elevation',
        vmax=4.0,
    )
