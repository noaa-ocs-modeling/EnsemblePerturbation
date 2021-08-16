from argparse import ArgumentParser

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cmocean
from matplotlib import pyplot
import pandas
from pandas import DataFrame
import tables

if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        'filename', help='filename (`*.h5`) generated by `combine_results`'
    )
    arguments = argument_parser.parse_args()

    combined_results_filename = arguments.filename

    input_dataframe: DataFrame = pandas.read_hdf(
        combined_results_filename, key='vortex_perturbation_parameters'
    )
    output_dataframe: DataFrame = pandas.read_hdf(combined_results_filename, key='zeta_max')

    combined_results = tables.open_file(combined_results_filename)

    # just get standard deviation of the outputs
    output_standard_deviation =output_dataframe.std(axis=0, skipna=True)

    # # Plot variable distributions the histogram of the data
    # for variable in input_dataframe:
    #     figure = pyplot.figure()
    #     axis = figure.add_subplot(1, 1, 1)
    #     axis.hist(input_dataframe[variable], bins='auto', density=True, alpha=0.75)
    #     axis.set_xlabel('Parameter')
    #     axis.set_ylabel('Probability')
    #     axis.set_title('Histogram of ' + variable)
    #     axis.grid(True)
    #     pyplot.show()

    # # plot histogram
    # figure = pyplot.figure()
    # axis = figure.add_subplot(1, 1, 1)
    # axis.hist(output_std, bins='auto', density=True, alpha=0.75)
    # axis.set_xlabel('Parameter')
    # axis.set_ylabel('Probability')
    # axis.set_title('Histogram of Maximum Elevation Variability')
    # axis.grid(True)
    # pyplot.show()

    # plot map
    figure = pyplot.figure(figsize=(18, 8))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    axis.scatter(
        output_dataframe.x,
        output_dataframe.y,
        s=1,
        c=output_standard_deviation,
        transform=ccrs.PlateCarree(),
        cmap=cmocean.cm.amp,
        vmin=0,
        vmax=1.6,
    )
    axis.coastlines()

    # pyplot.colorbar(ax=axis, shrink=0.98, extend='max', label='STD [m]')

    # Add the gridlines
    gridlines = axis.gridlines(color='black', linestyle='dotted', draw_labels=True, alpha=0.5)
    gridlines.top_labels = None
    gridlines.right_labels = None
    gridlines.xformatter = LONGITUDE_FORMATTER
    gridlines.yformatter = LATITUDE_FORMATTER

    figure.suptitle('Maximum Elevation Variability from 40-member ensemble')
    pyplot.show()
