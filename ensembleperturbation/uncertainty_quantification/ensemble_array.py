from os import PathLike

import numpy
import pandas
from pandas import DataFrame
from scipy.special import ndtri
from scipy.spatial import cKDTree

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

def sample_points_with_equal_spacing(output_dataframe: DataFrame, spacing: float):
    # get the mask indices for an equal spatial sampling of points
    
    # make the point tree
    points = output_dataframe[["x","y"]].to_numpy()
    point_tree = cKDTree(points)

    # enquire
    bad_indices = numpy.empty(0,dtype=int)
    good_indices = numpy.zeros(1,dtype=int)
    bad = 0
    counter = 0
    while bad < len(points)-1:
        counter = counter + 1
        idx_temp = point_tree.query_ball_point(points[good_indices[-1],:],spacing)
        idx_temp = numpy.setdiff1d(idx_temp,good_indices,assume_unique=True) 
        bad_indices = numpy.unique(numpy.append(bad_indices,idx_temp))
        for bdx,bad in enumerate(bad_indices):
            if bad <= good_indices[-1]:
               continue
            if bdx == len(bad_indices)-1 or bad_indices[bdx+1] - bad > 1:
               good_indices = numpy.append(good_indices,bad+1)
               break
    good_indices = good_indices[:-1]

    return good_indices    
