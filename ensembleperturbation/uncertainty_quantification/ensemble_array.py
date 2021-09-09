from os import PathLike

import numpy
import pandas
from pandas import DataFrame
from scipy.special import ndtri

def read_combined_hdf(
    filename: PathLike, input_key: str = None, output_key: str = None
) -> (DataFrame, DataFrame):
    if input_key is None:
        input_key = 'vortex_perturbation_parameters'
    if output_key is None:
        output_key = 'zeta_max'

    return (
        pandas.read_hdf(filename, input_key),
        pandas.read_hdf(filename, output_key),
    )


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

    # Transform the uniform dimension into gaussian
    for cdx,col in enumerate(input_dataframe.columns):
        if col == 'radius_of_maximum_winds': 
            pinput[:, cdx] = ndtri((pinput[:, cdx]+1.)/2.)

    print(f'Shape of parameter input: {pinput.shape}')
    print(f'Shape of model output: {output.shape}')

    return pinput, output
