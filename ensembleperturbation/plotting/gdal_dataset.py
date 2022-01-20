# from matplotlib import pyplot
# from matplotlib.axis import Axis
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize
import numpy

# from ensembleperturbation.plotting.geometry import plot_points
# import gdal
# from affine import Affine


def bounds_from_opposite_corners(
    corner_1: (float, float), corner_2: (float, float)
) -> (float, float, float, float):
    """
    get bounds from two XY points

    :param corner_1: XY point
    :param corner_2: XY point
    :returns: min X, min Y, max X, max Y
    """

    return numpy.ravel(numpy.sort(numpy.stack((corner_1, corner_2), axis=0), axis=0))


def where_not_nodata(array: numpy.ndarray, nodata: float = None) -> numpy.ndarray:
    """
    get a boolean array of where data exists in the given array

    :param array: array of gridded data with dimensions (Z)YX
    :param nodata: value where there is no data in the given array
    :returns: array of booleans indicating where data exists
    """

    if nodata is None:
        if array.dtype == bool:
            nodata = False
        elif array.dtype == int:
            nodata = -2147483648
        else:
            nodata = numpy.nan

    coverage = array != nodata if not numpy.isnan(nodata) else ~numpy.isnan(array)

    if len(array.shape) > 2:
        coverage = numpy.any(coverage, axis=0)

    return coverage


def _maxValue(arr: numpy.ndarray):
    """
    returns the most used value in the array as an integer

    Takes an input array and finds the most used value in the array, this
    value is used by the program to assume the array's nodata value

    :param arr: An input array
    """

    nums, counts = numpy.unique(arr, return_counts=True)
    index = numpy.where(counts == numpy.amax(counts))
    return int(nums[index])


def geoarray_to_xyz(
    data: numpy.ndarray,
    origin: (float, float),
    resolution: (float, float),
    nodata: float = None,
) -> numpy.ndarray:
    """
    extract XYZ points from an array of data using the given raster-like georeference (origin  and resolution)

    :param data: 2D array of gridded data
    :param origin: X, Y coordinates of northwest corner
    :param resolution: cell size
    :param nodata: value to exclude from point creation from the input grid
    :returns: N x 3 array of XYZ points
    """

    if nodata is None:
        if data.dtype == float:
            nodata = numpy.nan
        elif data.dtype == int:
            nodata = -2147483648
        if data.dtype == bool:
            nodata = False

    data_coverage = where_not_nodata(data, nodata)
    x_values, y_values = numpy.meshgrid(
        numpy.linspace(origin[0], origin[0] + resolution[0] * data.shape[1], data.shape[1]),
        numpy.linspace(origin[1], origin[1] + resolution[1] * data.shape[0], data.shape[0]),
    )

    return numpy.stack(
        (x_values[data_coverage], y_values[data_coverage], data[data_coverage]), axis=1
    )


# def gdal_to_xyz(dataset: gdal.Dataset, nodata: float = None) -> numpy.ndarray:
#     """
#     extract XYZ points from a GDAL dataset
#
#     :param dataset: GDAL dataset (point cloud or raster)
#     :param nodata: value to exclude from point creation
#     :returns: N x M array of XYZ points
#     """
#
#     coordinates = None
#     layers_data = []
#     if dataset.RasterCount > 0:
#         for index in range(1, dataset.RasterCount + 1):
#             raster_band = dataset.GetRasterBand(index)
#             geotransform = dataset.GetGeoTransform()
#
#             if nodata is None:
#                 nodata = raster_band.GetNoDataValue()
#
#             points = geoarray_to_xyz(
#                 raster_band.ReadAsArray(),
#                 origin=(geotransform[0], geotransform[3]),
#                 resolution=(geotransform[1], geotransform[5]),
#                 nodata=nodata,
#             )
#             if coordinates is None:
#                 coordinates = points[:, :2]
#             layers_data.append(points[:, 2])
#     else:
#         for index in range(dataset.GetLayerCount()):
#             point_layer = dataset.GetLayerByIndex(index)
#             num_features = point_layer.GetFeatureCount()
#
#             for feature_index in range(num_features):
#                 feature = point_layer.GetFeature(feature_index)
#                 feature_geometry = feature.geometry()
#                 num_points = feature_geometry.GetGeometryCount()
#                 # TODO this assumes points in all layers are in the same order
#                 points = numpy.array(
#                     [
#                         feature_geometry.GetGeometryRef(point_index).GetPoint()
#                         for point_index in range(num_points)
#                         if feature_geometry.GetGeometryRef(point_index).GetPoint()[2] != nodata
#                     ]
#                 )
#
#                 if coordinates is None:
#                     coordinates = points[:, :2]
#                 layers_data.append(points[:, 2])
#
#     return numpy.concatenate(
#         [coordinates] + [numpy.expand_dims(data, axis=1) for data in layers_data], axis=1
#     )
#
#
# def gdal_raster_bounds(raster: gdal.Dataset) -> (float, float, float, float):
#     """
#     get the bounds (grouped by dimension) of the given unrotated raster
#
#     :param raster: GDAL raster dataset
#     :returns: min X, min Y, max X, max Y
#     """
#
#     geotransform = raster.GetGeoTransform()
#     origin = numpy.array((geotransform[0], geotransform[3]))
#     resolution = numpy.array((geotransform[1], geotransform[5]))
#     rotation = numpy.array((geotransform[2], geotransform[4]))
#     shape = raster.RasterYSize, raster.RasterXSize
#
#     if numpy.any(rotation != 0):
#         raise NotImplementedError('rotated rasters not supported')
#
#     return bounds_from_opposite_corners(origin, origin + numpy.flip(shape) * resolution)
#
#
# def plot_geoarray(
#     array: numpy.ndarray,
#     transform: Affine = None,
#     nodata: float = None,
#     axis: Axis = None,
#     show: bool = False,
#     **kwargs,
# ):
#     """
#     plot the given georeferenced array
#
#     :param array: 2D array of gridded data
#     :param transform: affine matrix transform
#     :param nodata: value representing no data in the given data
#     :param axis: `pyplot` axis to plot to
#     :param show: whether to show the plot
#     """
#
#     origin = (transform.c, transform.f)
#     resolution = (transform.a, transform.e)
#
#     if nodata is None:
#         nodata = numpy.nan
#
#     if axis is None:
#         axis = pyplot.gca()
#
#     if not numpy.isnan(nodata):
#         array[~where_not_nodata(array, nodata)] = numpy.nan
#
#     # if resolution[0] < 0:
#     #     data = numpy.flip(data, axis=1)
#     # if resolution[1] < 0:
#     #     data = numpy.flip(data, axis=0)
#
#     bounds = bounds_from_opposite_corners(
#         origin, origin + numpy.flip(array.shape) * resolution
#     )
#     axis.matshow(array, extent=bounds[[0, 2, 1, 3]], aspect='auto', **kwargs)
#
#     if show:
#         pyplot.show()
#
#
# def plot_dataset(
#     dataset: gdal.Dataset, index: int = 0, axis: Axis = None, show: bool = False, **kwargs,
# ):
#     """
#     plot the given GDAL dataset.
#
#     :param dataset: GDAL dataset (raster or point cloud)
#     :param index: zero-based index of raster band / vector layer to read
#     :param axis: `pyplot` axis to plot to
#     :param show: whether to show the plot
#     """
#
#     if dataset.RasterCount > 0:
#         transform = Affine.from_gdal(*dataset.GetGeoTransform())
#
#         raster_band = dataset.GetRasterBand(index + 1)
#         raster_data = raster_band.ReadAsArray()
#         nodata = raster_band.GetNoDataValue()
#         if nodata is None:
#             nodata = _maxValue(raster_data)
#         del raster_band
#
#         plot_geoarray(raster_data.astype('float64'), transform, nodata, axis, show, **kwargs)
#     else:
#         plot_points(gdal_to_xyz(dataset), index, axis, show, **kwargs)
#
#
# def plot_interpolation(
#     original_dataset: gdal.Dataset,
#     interpolated_raster: gdal.Dataset,
#     method: str,
#     input_index: int = 0,
#     output_index: int = 0,
#     show: bool = False,
# ):
#     """
#     plot original data side-by-side with an interpolated raster for comparison
#
#     :param original_dataset: GDAL dataset (point cloud or raster) of original data
#     :param interpolated_raster: GDAL raster of interpolated data
#     :param method: method of interpolation
#     :param input_index: zero-based index of layer / band to read from the input dataset
#     :param output_index: zero-based index of band to read from the output raster
#     :param show: whether to show the plot
#     """
#
#     if original_dataset.RasterCount > 0:
#         original_raster_band = original_dataset.GetRasterBand(input_index + 1)
#         original_data = original_raster_band.ReadAsArray()
#         original_nodata = original_raster_band.GetNoDataValue()
#         original_data[original_data == original_nodata] = numpy.nan
#     else:
#         original_data = gdal_to_xyz(original_dataset)[:, input_index + 2]
#
#     interpolated_raster_band = interpolated_raster.GetRasterBand(output_index + 1)
#     interpolated_nodata = interpolated_raster_band.GetNoDataValue()
#     interpolated_data = interpolated_raster_band.ReadAsArray()
#     interpolated_data[interpolated_data == interpolated_nodata] = numpy.nan
#
#     # get minimum and maximum values for all three dimensions
#     z_values = numpy.concatenate((numpy.ravel(interpolated_data), numpy.ravel(original_data)))
#     min_z = numpy.nanmin(z_values)
#     max_z = numpy.nanmax(z_values)
#
#     # create a new figure window with two subplots
#     figure = pyplot.figure()
#     left_axis = figure.add_subplot(1, 2, 1)
#     left_axis.set_title('survey data')
#     right_axis = figure.add_subplot(1, 2, 2, sharex=left_axis, sharey=left_axis)
#     right_axis.set_title(f'{method} interpolation to raster')
#
#     # plot data
#     plot_dataset(original_dataset, input_index, left_axis, vmin=min_z, vmax=max_z)
#     plot_dataset(interpolated_raster, input_index, right_axis, vmin=min_z, vmax=max_z)
#     right_axis.axes.get_yaxis().set_visible(False)
#
#     # create colorbar
#     figure.colorbar(
#         ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z)), ax=(right_axis, left_axis)
#     )
#
#     # pause program execution and show the figure
#     if show:
#         pyplot.show()
