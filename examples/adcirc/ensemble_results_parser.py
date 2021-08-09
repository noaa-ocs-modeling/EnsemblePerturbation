from pathlib import Path

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.perturbation.atcf import parse_vortex_perturbations
from ensembleperturbation.utilities import get_logger

"""
For a particular set of ensemble runs, 
read JSON vortex pertubation input parameter files,
and the ADCIRC maxele.63.nc output results in a specific subdomain.
Assemble these into pandas dataframes (matrix-like structure)
and save into a single HDF5 file

Author: William Pringle
Date:   July 2021
"""

LOGGER = get_logger('combine_results')

if __name__ == '__main__':
    # get the directory we want to use in a "Path-like" format
    input_directory = Path.cwd()
    track_directory = input_directory / 'track_files'
    runs_directory = input_directory / 'runs'

    # get the input parameters
    vortex_perturbations = parse_vortex_perturbations(
        track_directory, output_filename='vortices.h5'
    )
    print(vortex_perturbations)

    # define the subdomain to extract
    maximum_depth = 250
    bounds = (-81.0, 32.0, -75.0, 36.5)

    # get the output values
    output_dataframe = combine_outputs(
        runs_directory, maximum_depth=maximum_depth, bounds=bounds, output_filename='runs.h5'
    )
    print(output_dataframe)

    print('done')
