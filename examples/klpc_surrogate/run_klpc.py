#!/usr/bin/env python

import glob as gl
import os
import shutil

from matplotlib import pyplot
import numpy as np

from ensembleperturbation.plotting import plot_points
from ensembleperturbation.uncertainty_quantification.ensemble_array import (
    ensemble_array,
    read_combined_hdf,
)
from ensembleperturbation.uncertainty_quantification.karhunen_loeve_expansion import (
    karhunen_loeve_expansion,
    karhunen_loeve_pc_coefficients,
    karhunen_loeve_prediction,
)
from ensembleperturbation.uncertainty_quantification.polynomial_chaos import (
    build_pc_expansion,
    evaluate_pc_distribution_function,
    evaluate_pc_exceedance_heights,
    evaluate_pc_exceedance_probabilities,
    evaluate_pc_expansion,
    evaluate_pc_multiindex,
    evaluate_pc_sensitivity,
)


def joint_klpc_surrogate(
    h5name: str,
    point_spacing: int,
    neig: float,
    pc_type: str,
    lambda_reg: int,
    pc_order: int,
    mi_type: str,
    num_samples: int,
    pdf_bins: int,
    exceedance_probabilities: np.ndarray,
    exceedance_heights: np.ndarray,
):

    # ------------------------#
    # -------- Part 1 --------#
    # ------------------------#
    print('Reading ensemble inputs and outputs...')

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
    np_output_subset = np_output[:, ::point_spacing]
    points_subset = points[::point_spacing, :]

    # alternative way get points (leave for now)
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

    pc_dim = np_input.shape[1]  # dimension of the PC expansion
    ngrid = ymodel.shape[0]  # number of points
    nens = ymodel.shape[1]  # number of ensembles
    print(f'Reduced grid size = {ngrid}')

    # ------------------------#
    # -------- Part 2 --------#
    # ------------------------#
    print('Evaluating the Karhunen-Loeve expansion...')

    ## Evaluating the KL mode
    # Components of the dictionary:
    # mean_vector is the average field                                        : size (ngrid,)
    # modes is the KL modes ('principal directions')                          : size (ngrid,neig)
    # eigenvalues is the eigenvalue vector                                    : size (neig,)
    # samples are the samples for the KL coefficients                         : size (nens, neig)
    kl_dict = karhunen_loeve_expansion(ymodel, neig=neig, plot=True)

    neig = len(kl_dict['eigenvalues'])  # number of eigenvalues
    print(f'Number of KL modes = {neig}')

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
            vmin=0.0,
        )

        # plot_coastline()
        plot_points(
            np.hstack((points_subset, ypred[:, [example]])),
            save_filename='predicted_zmax' + str(example),
            title='predicted zmax, ensemble #' + str(example),
            vmax=3.0,
            vmin=0.0,
        )

    # ------------------------#
    # -------- Part 3 --------#
    # ------------------------#
    print('Evaluating the Polynomial Chaos expansion for each KL mode...')

    # find the multi-index file for later use
    mi_filename = 'mindex.dat'
    evaluate_pc_multiindex(
        multiindex_filename=mi_filename,
        multiindex_type=mi_type,
        pc_dimension=pc_dim,
        poly_order=pc_order,
    )

    # Build PC for each KL mode (each mode has nens values)
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
            ncoefs = len(poly_coefs)  # number of polynomial coefficients
            pc_coefficients = np.empty(
                (neig, ncoefs)
            )  # array for storing PC coefficients for each KL mode
        pc_coefficients[mode, :] = poly_coefs

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

    # ------------------------#
    # -------- Part 4 --------#
    # ------------------------#
    print('Evaluating the joint KLPC expansion at each point...')

    # Now get the joint KLPC coefficients
    klpc_coefficients = karhunen_loeve_pc_coefficients(
        kl_dict=kl_dict, pc_coefficients=pc_coefficients,
    )

    # Get sensitivities and distribution functions
    # at each point of the KLP surrgogate
    param_filename = 'coeff.dat'
    klpc_sensitivities = [None] * ngrid
    klpc_exceedance_heights = [None] * ngrid
    klpc_exceedance_probabilities = [None] * ngrid
    for z_index in range(ngrid):
        # save coefficients for this point into parameter file
        np.savetxt(param_filename, klpc_coefficients[z_index, :])

        # evaluate the Sobol sensitivities for the KLPC surrogate
        klpc_sensitivities[z_index] = evaluate_pc_sensitivity(
            parameter_filename=param_filename,
            multiindex_filename=mi_filename,
            pc_type=pc_type,
        )

        # evaluate the PDF/CDF of the KLPC surrogate
        klpc_distribution = evaluate_pc_distribution_function(
            parameter_filename=param_filename,
            multiindex_filename=mi_filename,
            num_samples=num_samples,
            pdf_bins=pdf_bins,
            pc_type=pc_type,
            pc_dimension=pc_dim,
            poly_order=pc_order,
        )

        # Get the heights at the desired exceedance probabilities
        klpc_exceedance_heights[z_index] = evaluate_pc_exceedance_heights(
            exceedance_probabilities=exceedance_probabilities, pc_dict=klpc_distribution,
        )

        # Get the probability of the desired exceedance height
        klpc_exceedance_probabilities[z_index] = evaluate_pc_exceedance_probabilities(
            exceedance_heights=exceedance_heights, pc_dict=klpc_distribution,
        )

    # ------------------------#
    # -------- Part 5 --------#
    # ------------------------#
    print('Plotting sensitivities and exceedance levels...')
    # Plot scatter points of sensitivities
    variable_names = dataframes[keys[0]].columns
    sensitivity_types = klpc_sensitivities[0].keys()
    for sensitivity_type in sensitivity_types:
        print('Plotting ' + sensitivity_type + ' sensitivities')
        sensitivity = np.array([z[sensitivity_type] for z in klpc_sensitivities])
        for vdx1, var1 in enumerate(variable_names):
            if sensitivity.ndim == 2:
                plot_points(
                    np.hstack((points_subset, sensitivity[:, [vdx1]])),
                    save_filename=var1 + '_' + sensitivity_type + '_sensitivity',
                    title=sensitivity_type + ' sensitivity for ' + var1,
                    vmax=1.0,
                )
            else:
                # Don't understand joint sensitivity matrix format..
                for vdx2, var2 in enumerate(variable_names):
                    plot_points(
                        np.hstack((points_subset, sensitivity[:, [vdx1], [vdx2]])),
                        save_filename=var1
                        + '+'
                        + var2
                        + '_'
                        + sensitivity_type
                        + '_sensitivity',
                        title=sensitivity_type + ' sensitivity for ' + var1 + '/' + var2,
                        vmax=1.0,
                    )

    # Plot scatter points of exceedance levels for each probability
    for pdx, probability in enumerate(exceedance_probabilities):
        label = str(int(probability * 100)) + '% exceedance height'
        filename = 'height_of_' + str(int(probability * 100)) + 'percent_exceedance'
        print('Plotting ' + label)
        height = np.array([z[pdx] for z in klpc_exceedance_heights])
        plot_points(
            np.hstack((points_subset, height[:, None])),
            save_filename=filename,
            title=label,
            vmax=3.0,
            vmin=0.0,
        )

    # Plot scatter points of exceedance probability for each height
    for hdx, height in enumerate(exceedance_heights):
        label = str(height) + '-m exceedance probability'
        filename = 'probability_of_' + str(height) + 'm_exceedance.png'
        print('Plotting ' + label)
        probability = np.array([z[hdx] for z in klpc_exceedance_probabilities])
        plot_points(
            np.hstack((points_subset, probability[:, None])),
            save_filename=filename,
            title=label,
            vmax=1.0,
        )


#############################
####### MAIN SCRIPT #########
#############################
if __name__ == '__main__':

    # Ensemble member data and point selection
    h5name = '../data/florence_40member.h5'  # name of h5 data
    point_spacing = 25  # select every x points

    # Karhunen Loeve expansion parameters
    neig = 0.95  # choose neig decimal percent of variance explained to keep

    # Polynomial Chaos expansion parameters
    pc_type = 'HG'  # Hermite-Gauss chaos
    lambda_reg = 0  # regularization lambda
    pc_order = 3  # order of the polynomial
    mi_type = 'TO'  # PC multi-index type

    # Probability parameters
    num_samples = 1000  # number of times to sample the KLPC expansion to get PDF
    pdf_bins = 1000  # number of PDF bins
    exceedance_probabilities = np.array([0.1, 0.5, 0.9])  # [decimal percent]
    exceedance_heights = np.array([0.5, 1.0, 2.0])  # [m]

    # ------------------------------------------------#
    # Execute the joint klpc program                 #
    # ------------------------------------------------#
    joint_klpc_surrogate(
        h5name=h5name,
        point_spacing=point_spacing,
        neig=neig,
        pc_type=pc_type,
        lambda_reg=lambda_reg,
        pc_order=pc_order,
        mi_type=mi_type,
        num_samples=num_samples,
        pdf_bins=pdf_bins,
        exceedance_probabilities=exceedance_probabilities,
        exceedance_heights=exceedance_heights,
    )

    # ------------------------------------------------#
    # Move files and figures into output directories #
    # ------------------------------------------------#
    dst_folder = 'data/'
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)
    for data in gl.glob('*.dat') + gl.glob('*.log'):
        file_name = os.path.basename(data)
        shutil.move(data, dst_folder + file_name)

    dst_folder = 'figures/'
    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)
    for figure in gl.glob('*.png'):
        file_name = os.path.basename(figure)
        shutil.move(figure, dst_folder + file_name)
