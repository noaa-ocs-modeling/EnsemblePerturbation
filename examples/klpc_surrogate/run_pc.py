import os
import sys

import matplotlib.pyplot as pyplot
import numpy as np
from scipy.special import ndtri

from ensembleperturbation.uncertainty_quantification.ensemble_array import (
    ensemble_array,
    read_combined_hdf,
)

if __name__ == '__main__':
    plot = False

    # if plot:
    #     x = np.linspace(-1.,1.,1000)
    #     y = ndtri((x+1.)/2.)
    #     pyplot.plot(x,y)
    #     pyplot.show()
    #     sys.exit()

    # Load the input
    pinput, output = ensemble_array(
        *read_combined_hdf(
            filename=r'run_20210812_florence_multivariate_besttrack_250msubset_40members.h5'
        )
    )

    # Transform the uniform dimension into gaussian
    pinput[:, 2] = ndtri((pinput[:, 2] + 1.0) / 2.0)

    output = np.loadtxt('output.txt')

    output = np.nan_to_num(output)

    # Pick a QoI of interest, for example, the mean of the whole region
    qoi = np.mean(output, axis=1)

    np.savetxt('qoi.txt', qoi)
    # Builds second order PC expansion for the QoI
    uqtk_cmd = 'regression -x pinput.txt -y qoi.txt -s HG -o 2 -l 0'
    os.system(uqtk_cmd)

    np.savetxt('xdata.dat', pinput)  # because pce_eval expects xdata.dat as input
    # Evaluates the constructed PC at the input for comparison
    uqtk_cmd = 'pce_eval -f coeff.dat -s HG -o 2'
    os.system(uqtk_cmd)
    qoi_pc = np.loadtxt('ydata.dat')

    if plot:
        pyplot.plot(qoi, qoi_pc, 'o')
        pyplot.plot([0, 1], [0, 1], 'k--', lw=1)
        pyplot.show()
