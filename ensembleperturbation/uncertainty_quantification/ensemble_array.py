from os import PathLike

import numpy
import pandas
from pandas import DataFrame
import tables


def read_combined_hdf(filename: PathLike) -> {str: DataFrame}:
    keys = [group._v_name for group in tables.open_file(filename).walk_groups('/')]
    return {key: pandas.read_hdf(filename, key) for key in keys if key != '/'}


def ensemble_array(
    input_dataframe: DataFrame, output_dataframe: DataFrame
) -> (numpy.ndarray, numpy.ndarray):
    nens = len(input_dataframe)
    ngrid = len(output_dataframe)
    dim = 4

    print(f'Parameter dimensionality: {dim}')
    print(f'Ensemble size: {nens}')
    print(f'Spatial grid size: {ngrid}')

    # Convert to proper numpy (there must be a cute pandas command to do this in a line or two...)
    pinput = numpy.empty((0, dim))
    output = numpy.empty((0, ngrid))
    for iens in range(nens):
        sample_key = 'vortex_4_variable_perturbation_' + str(iens + 1)
        pinput = numpy.append(
            pinput, input_dataframe.loc[sample_key + '.json'].to_numpy().reshape(1, -1), axis=0
        )
        output = numpy.append(
            output, output_dataframe[sample_key].to_numpy().reshape(1, -1), axis=0
        )

    print(f'Shape of parameter input: {pinput.shape}')
    print(f'Shape of model output: {output.shape}')

    return pinput, output
