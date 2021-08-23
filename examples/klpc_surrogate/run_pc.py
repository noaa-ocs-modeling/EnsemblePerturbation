#!/usr/bin/env python

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri

# x = np.linspace(-1.,1.,1000)
# y = ndtri((x+1.)/2.)
# plt.plot(x,y)
# plt.show()
# sys.exit()

# Load the input
pinput = np.loadtxt('pinput.txt')
# Transform the uniform dimension into gaussian
pinput[:, 2] = ndtri((pinput[:, 2]+1.)/2.)

output = np.loadtxt('output.txt')

output = np.nan_to_num(output)

# Pick a QoI of interest, for example, the mean of the whole region
qoi = np.mean(output, axis=1)



np.savetxt('qoi.txt', qoi)
# Builds second order PC expansion for the QoI
uqtk_cmd = 'regression -x pinput.txt -y qoi.txt -s HG -o 2 -l 0'
os.system(uqtk_cmd)

np.savetxt('xdata.dat', pinput) #because pce_eval expects xdata.dat as input
# Evaluates the constructed PC at the input for comparison
uqtk_cmd = 'pce_eval -f coeff.dat -s HG -o 2'
os.system(uqtk_cmd)
qoi_pc = np.loadtxt('ydata.dat')

plt.plot(qoi, qoi_pc, 'o')
plt.plot([0,1],[0,1], 'k--', lw=1)
plt.show()
