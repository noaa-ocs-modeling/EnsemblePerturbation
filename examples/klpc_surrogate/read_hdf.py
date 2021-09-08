import cartopy
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cmocean
from matplotlib import pyplot
import numpy

from ensembleperturbation.uncertainty_quantification.ensemble_array import (
    ensemble_array,
    read_combined_hdf,
)

if __name__ == '__main__':
    input_filename = 'run_20210812_florence_multivariate_besttrack_250msubset_40members.h5'
    input_dataframe, output_dataframe = read_combined_hdf(input_filename=input_filename)

    pinput, output = ensemble_array(
        input_dataframe=input_dataframe, output_dataframe=output_dataframe,
    )

    # somehow this gives slightly different answer compared to pandas
    output_standard_deviation = numpy.nanstd(output, axis=0)
    # output_standard_deviation = output_dataframe.filter(regex='vortex*', axis=1).std(axis=1, skipna=True)

    numpy.savetxt('pinput.txt', pinput)
    numpy.savetxt('pinput.txt', output)
    numpy.savetxt('output_std.txt', output_standard_deviation)

    # plotting the histogram of standard deviation (but I am not sure what is this helpful for frankly)
    output_standard_deviation_clean = output_standard_deviation[
        ~numpy.isnan(output_standard_deviation) & output_standard_deviation != 0
    ]
    pyplot.hist(output_standard_deviation_clean, bins='auto', density=True, alpha=0.75)
    pyplot.xlabel('Parameter')
    pyplot.ylabel('Probability')
    pyplot.title('Histogram of Maximum Elevation Variability')
    pyplot.grid(True)
    pyplot.savefig('hist_maxelevstd.png')
    pyplot.clf()

    # plot map
    nevery = 1
    fig = pyplot.figure(figsize=(12, 8))
    ax = pyplot.axes(projection=cartopy.crs.PlateCarree())
    pyplot.scatter(
        output_dataframe['x'][:nevery],
        output_dataframe['y'][:nevery],
        s=1,
        c=output_standard_deviation[::nevery],
        transform=cartopy.crs.PlateCarree(),
        cmap=cmocean.cm.amp,
        vmin=0,
        vmax=1.6,
    )
    pyplot.colorbar(ax=ax, shrink=0.98, extend='max', label='STD [m]')
    ax.coastlines()
    # Add the gridlines
    gl = ax.gridlines(color='black', linestyle='dotted', draw_labels=True, alpha=0.5)
    gl.top_labels = None
    gl.right_labels = None
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    pyplot.title('Maximum Elevation Variability from 40-member ensemble')
    pyplot.savefig('map_maxelevstd.png')
    pyplot.clf()
