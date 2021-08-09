import json
from os import PathLike
from pathlib import Path

import pandas

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


def parse_vortex_perturbations(directory: PathLike = None, write_to_file: bool = False):
    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    # reading the JSON data using json.load()
    perturbations = {}
    for filename in directory.glob('**/vortex*.json'):
        with open(filename) as vortex_file:
            perturbations[filename.stem] = json.load(vortex_file)

    if len(perturbations) == 0:
        raise FileNotFoundError(
            f'could not find any perturbation JSON file(s) in "{directory}"'
        )

    # convert dictionary to dataframe with:
    # rows -> name of perturbation
    # columns -> name of each variable that is perturbed
    # values -> the perturbation parameter for each variable
    perturbations = {
        vortex: perturbations[vortex]
        for vortex, number in sorted(
            {
                vortex: int(vortex.split('_')[-1])
                for vortex, pertubations in perturbations.items()
            }.items(),
            key=lambda item: item[1],
        )
    }

    perturbations = pandas.DataFrame.from_records(
        list(perturbations.values()), index=list(perturbations)
    )

    if write_to_file:
        perturbations.to_hdf(
            directory.stem + '.h5',
            key='vortex_perturbation_parameters',
            mode='w',
            format='table',
            data_columns=True,
        )

    return perturbations


def parse_output(
    directory: PathLike = None,
    bounds: (float, float, float, float) = None,
    maximum_depth: float = None,
    write_to_file: bool = False,
):
    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    # define the output file type and variable name interested name
    output_filetypes = {
        'maxele.63.nc': 'zeta_max',
    }

    # parse all the outputs using built-in parser
    output_data = parse_adcirc_outputs(
        directory=directory, file_data_variables=output_filetypes.keys(),
    )

    if len(output_data) == 0:
        raise FileNotFoundError(f'could not find any output files in "{directory}"')

    # now assemble results into a single dataframe with:
    # rows -> index of a vertex in the mesh subset
    # columns -> name of perturbation, ( + x, y (lon, lat) and depth info)
    # values -> maximum elevation values ( + location and depths)
    subset = None
    dataframe = None
    for pertubation_index, perturbation in enumerate(output_data):
        for variable in output_data[perturbation]:
            variable_dataframe = output_data[perturbation][variable]
            if subset is None:
                subset = pandas.Series(True, index=variable_dataframe.index)
                if maximum_depth is not None:
                    subset &= variable_dataframe['depth'] < maximum_depth
                if bounds is not None:
                    subset &= (variable_dataframe['x'] > bounds[0]) & (variable_dataframe['x'] < bounds[2])
                    subset &= (variable_dataframe['y'] > bounds[1]) & (variable_dataframe['y'] < bounds[3])
                dataframe = variable_dataframe[['x', 'y', 'depth']][subset]
            dataframe.insert(2, perturbation, variable_dataframe[output_filetypes[variable]][subset], True)

    if write_to_file:
        dataframe.to_hdf(
            directory.name + '.h5',
            key=output_filetypes[variable],
            mode='a',
            format='table',
            data_columns=True,
        )

    return dataframe


if __name__ == '__main__':
    # get the directory we want to use in a "Path-like" format
    input_directory = Path.cwd()
    track_directory = input_directory / 'track_files'
    runs_directory = input_directory / 'runs'

    # get the input parameters
    vortex_perturbations = parse_vortex_perturbations(track_directory, write_to_file=False)
    print(vortex_perturbations)

    # define the subdomain to extract
    maximum_depth = 250
    bounds = (-81.0, 32.0, -75.0, 36.5)

    # get the output values
    output_dataframe = parse_output(
        runs_directory, maximum_depth=maximum_depth, bounds=bounds, write_to_file=False
    )
    print(output_dataframe)

    print('done')
