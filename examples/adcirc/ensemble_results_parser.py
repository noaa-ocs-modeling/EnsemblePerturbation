from json import load
from os import getcwd
from pathlib import Path

import pandas as pd

from ensembleperturbation.parsing.adcirc import parse_adcirc_outputs

"""
For a particular set of ensemble runs, 
read JSON vortex pertubation input parameter files,
and the ADCIRC maxele.63.nc output results in a specific subdomain.
Assemble these into pandas dataframes (matrix-like structure)
and save into a single HDF5 file

Author: William Pringle
Date:   July 2021
"""


def parse_input(input_directory):
    # reading the JSON data using json.load()
    # create the dictionary KEYS
    dict_vortex = {}
    for filename in input_directory.glob('**/vortex.json'):
        with open(filename) as vortex_file:
            dict_temp = load(vortex_file)
            dict_vortex.update(dict_temp)
    for key in dict_vortex:
        dict_vortex[key] = []

    # Fill in the dictionary for each perturbation
    indices = []
    for filename in input_directory.glob('**/vortex.json'):
        # get the perturbation index
        indices.append(filename.parts[-2])
        with open(filename) as vortex_file:
            dict_temp = load(vortex_file)
            for key in dict_vortex:
                try:
                    dict_vortex[key].append(dict_temp[key])
                except:
                    dict_vortex[key].append(0.0)

    # convert dictionary to dataframe with:
    # rows -> name of perturbation
    # columns -> name of each variable that is perturbed
    # values -> the perturbation parameter for each variable
    ds_vortex = pd.DataFrame.from_dict(dict_vortex)
    ds_vortex.index = indices
    print(ds_vortex)
    ds_vortex.to_hdf(
        input_directory.name + '.h5',
        key='vortex_perturbation_parameters',
        mode='w',
        format='table',
        data_columns=True,
    )


def parse_output(input_directory):
    # define the output file type and variable name interested name
    output_filetypes = {
        'maxele.63.nc': 'zeta_max',
    }
    # define the subdomain to extract
    max_depth = 250.0  # maximum depth to care about
    lon_min = -81.0  # minimum longitude to care about
    lon_max = -75.0  # maximum longitude to care about
    lat_min = +32.0  # minimum latitude to care about
    lat_max = +36.5  # maximum latitude to care about

    # parse all the outputs using built-in parser
    output_data = parse_adcirc_outputs(
        directory=input_directory, file_data_variables=output_filetypes.keys(),
    )

    # now assemble results into a single dataframe with:
    # rows -> index of a vertex in the mesh subset
    # columns -> name of perturbation, ( + x, y (lon, lat) and depth info)
    # values -> maximum elevation values ( + location and depths)
    for pdx, perturbation in enumerate(output_data):
        for vdx, variable in enumerate(output_data[perturbation]):
            ds = output_data[perturbation][variable]
            if pdx == 0:
                depth = ds['depth']
                lon = ds['x']
                lat = ds['y']
                subset = (
                    (depth < max_depth)
                    & (lon > lon_min)
                    & (lon < lon_max)
                    & (lat > lat_min)
                    & (lat < lat_max)
                )
                ds_subset = ds[['x', 'y', 'depth']][subset]
            ds_temp = ds[output_filetypes[variable]][subset]
            ds_subset.insert(2, perturbation, ds_temp, True)

    print(ds_subset)
    ds_subset.to_hdf(
        input_directory.name + '.h5',
        key=output_filetypes[variable],
        mode='a',
        format='table',
        data_columns=True,
    )


if __name__ == '__main__':
    # get the directory we want to use in a "Path-like" format
    directory = getcwd()
    directory = Path(directory)
    # get the input parameters
    parse_input(directory)
    # get the output values
    parse_output(directory)
