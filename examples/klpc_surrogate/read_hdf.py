#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import cmocean
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt


h5file = 'run_20210812_florence_multivariate_besttrack_250msubset_40members.h5'

input_key = 'vortex_perturbation_parameters'
input_df = pd.read_hdf(h5file,input_key)

output_key = 'zeta_max'
output_df = pd.read_hdf(h5file,output_key)

lons = output_df['x']
lats = output_df['y']
lons_lats_list = list(zip(lons, lats))



nens = len(input_df)
ngrid = len(lons_lats_list)
dim = 4
print('Parameter dimensionality is ', dim)
print('Ensemble size is ', nens)
print('Spatial grid size is ', ngrid)

# Convert to proper numpy (there must be a cute pandas command to do this in a line or two...)
pinput = np.empty((0, dim))
output = np.empty((0, ngrid))
for iens in range(nens):
    sample_key = 'vortex_4_variable_perturbation_'+str(iens+1)
    pinput = np.append(pinput, input_df.loc[sample_key+'.json'].to_numpy().reshape(1, -1), axis=0)
    output = np.append(output, output_df[sample_key].to_numpy().reshape(1, -1), axis=0)

print('Shape of parameter input is ', pinput.shape)
print('Shape of model output is ', output.shape)

np.savetxt('pinput.txt', pinput)
np.savetxt('output.txt', output)

output_std= np.nanstd(output, axis=0)
# somehow this gives slightly different answer compared to pandas
# output_std = output_df.filter(regex='vortex*', axis=1).std(axis=1,skipna=True)
np.savetxt('output_std.txt', output_std)

# cleanup before histogram
oo = output_std[~np.isnan(output_std)]
output_std_clean = oo[np.nonzero(oo)]
# plotting the histogram of standard deviation (but I am not sure what is this helpful for frankly)
plt.hist(output_std_clean, bins='auto', density=True, alpha=0.75)
plt.xlabel('Parameter')
plt.ylabel('Probability')
plt.title('Histogram of Maximum Elevation Variability')
plt.grid(True)
plt.savefig('hist_maxelevstd.png')
plt.clf()

# plot map
nevery = 1
fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.scatter(lons[::nevery], lats[::nevery],
            s=1,
            c=output_std[::nevery],
            transform=ccrs.PlateCarree(),
            cmap=cmocean.cm.amp,
            vmin=0,vmax=1.6
)
plt.colorbar(ax=ax, shrink=.98,extend='max',label='STD [m]')
ax.coastlines()
# Add the gridlines
gl = ax.gridlines(color="black", linestyle="dotted", draw_labels=True, alpha=0.5)
gl.top_labels = None
gl.right_labels = None
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.title("Maximum Elevation Variability from 40-member ensemble")
plt.savefig('map_maxelevstd.png')
plt.clf()
