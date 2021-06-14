import io
from os import PathLike
import pathlib
from typing import Union
import zipfile

# from affine import Affine
import appdirs

# import gdal
import geopandas
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
import numpy
import requests
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape

from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('plotting')


# def geoarray_to_xyz(
#     data: numpy.array, origin: (float, float), resolution: (float, float), nodata: float = None
# ) -> numpy.array:
#     """
#     Extract XYZ points from an array of data using the given raster-like
#     georeference (origin  and resolution).
#
#     Parameters
#     ----------
#     data
#         2D array of gridded data
#     origin
#         X, Y coordinates of northwest corner
#     resolution
#         cell size
#     nodata
#         value to exclude from point creation from the input grid
#
#     Returns
#     -------
#     numpy.array
#         N x 3 array of XYZ points
#     """
#
#     if nodata is None:
#         if data.dtype == float:
#             nodata = numpy.nan
#         elif data.dtype == int:
#             nodata = -2147483648
#         if data.dtype == bool:
#             nodata = False
#
#     data_coverage = where_not_nodata(data, nodata)
#     x_values, y_values = numpy.meshgrid(
#         numpy.linspace(origin[0], origin[0] + resolution[0] * data.shape[1], data.shape[1]),
#         numpy.linspace(origin[1], origin[1] + resolution[1] * data.shape[0], data.shape[0]),
#     )
#
#     return numpy.stack(
#         (x_values[data_coverage], y_values[data_coverage], data[data_coverage]), axis=1
#     )
#
#
# def gdal_to_xyz(dataset: gdal.Dataset, nodata: float = None) -> numpy.array:
#     """
#     Extract XYZ points from a GDAL dataset.
#
#     Parameters
#     ----------
#     dataset
#         GDAL dataset (point cloud or raster)
#     nodata
#         value to exclude from point creation
#
#     Returns
#     -------
#     numpy.array
#         N x M array of XYZ points
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


def bounds_from_opposite_corners(
    corner_1: (float, float), corner_2: (float, float)
) -> (float, float, float, float):
    """
    Get bounds from two XY points.

    Parameters
    ----------
    corner_1
        XY point
    corner_2
        XY point

    Returns
    -------
    float, float, float, float
        min X, min Y, max X, max Y
    """

    return numpy.ravel(numpy.sort(numpy.stack((corner_1, corner_2), axis=0), axis=0))


# def gdal_raster_bounds(raster: gdal.Dataset) -> (float, float, float, float):
#     """
#     Get the bounds (grouped by dimension) of the given unrotated raster.
#
#     Parameters
#     ----------
#     raster
#         GDAL raster dataset
#
#     Returns
#     -------
#     float, float, float, float
#         min X, min Y, max X, max Y
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


def where_not_nodata(array: numpy.array, nodata: float = None) -> numpy.array:
    """
    Get a boolean array of where data exists in the given array.

    Parameters
    ----------
    array
        array of gridded data with dimensions (Z)YX
    nodata
        value where there is no data in the given array

    Returns
    -------
    numpy.array
        array of booleans indicating where data exists
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


def plot_polygon(
    geometry: Union[Polygon, MultiPolygon],
    axis: pyplot.Axes = None,
    show: bool = False,
    **kwargs
):
    """
    Plot the given polygon.

    Parameters
    ----------
    geometry
        Shapely polygon (or multipolygon)
    axis
        `pyplot` axis to plot to
    show
        whether to show the plot
    """

    if axis is None:
        axis = pyplot.gca()

    if 'c' not in kwargs:
        try:
            color = next(axis._get_lines.color_cycle)
        except AttributeError:
            color = 'r'
        kwargs['c'] = color

    if isinstance(geometry, dict):
        geometry = shapely_shape(geometry)

    if type(geometry) is Polygon:
        axis.plot(*geometry.exterior.xy, **kwargs)
        for interior in geometry.interiors:
            axis.plot(*interior.xy, **kwargs)
    elif type(geometry) is MultiPolygon:
        for polygon in geometry:
            plot_polygon(polygon, axis, show=False, **kwargs)
    else:
        axis.plot(*geometry.xy, **kwargs)

    if show:
        pyplot.show()


def plot_polygons(
    geometries: [Polygon],
    colors: [str] = None,
    axis: pyplot.Axes = None,
    show: bool = False,
    **kwargs
):
    """
    Plot the given polygons using the given colors.

    Parameters
    ----------
    geometries
        list of shapely polygons or multipolygons
    colors
        colors to plot each region
    axis
        `pyplot` axis to plot to
    show
        whether to show the plot
    """

    if axis is None:
        axis = pyplot.gca()

    if 'c' in kwargs:
        colors = [kwargs['c'] for _ in range(len(geometries))]
    elif colors is None:
        colors = [
            get_cmap('gist_rainbow')(color_index / len(geometries))
            for color_index in range(len(geometries))
        ]

    for geometry_index, geometry in enumerate(geometries):
        kwargs['c'] = colors[geometry_index]
        plot_polygon(geometry, axis, **kwargs)

    if show:
        pyplot.show()


def plot_bounding_box(
    sw: (float, float),
    ne: (float, float),
    axis: pyplot.Axes = None,
    show: bool = False,
    **kwargs
):
    """
    Plot the bounding box of the given extent.

    Parameters
    ----------
    sw
        XY coordinates of southwest corner
    ne
        XY coordinates of northeast corner
    axis
        `pyplot` axis to plot to
    show
        whether to show the plot
    """

    if axis is None:
        axis = pyplot.gca()

    corner_points = numpy.array([sw, (ne[0], sw[1]), ne, (sw[0], ne[1]), sw])

    axis.plot(corner_points[:, 0], corner_points[:, 1], **kwargs)

    if show:
        pyplot.show()


def plot_points(
    points: Union[numpy.array, MultiPoint],
    index: int = 0,
    axis: pyplot.Axes = None,
    show: bool = False,
    **kwargs
):
    """
    Create a scatter plot of the given points.

    Parameters
    ----------
    points
        N x M array of points
    index
        zero-based index of vector layer to read
    axis
        `pyplot` axis to plot to
    show
        whether to show the plot
    """

    if type(points) is MultiPoint:
        points = numpy.squeeze(numpy.stack((point._get_coords() for point in points), axis=0))

    if axis is None:
        axis = pyplot.gca()

    if 'c' not in kwargs and points.shape[1] > 2:
        kwargs['c'] = points[:, index + 2]

    if 's' not in kwargs:
        kwargs['s'] = 2

    axis.scatter(points[:, 0], points[:, 1], **kwargs)

    if show:
        pyplot.show()


# def plot_geoarray(array: numpy.array, transform: Affine = None,
#                   nodata: float = None, axis: pyplot.Axes = None,
#                   show: bool = False, **kwargs):
#     """
#     Plot the given georeferenced array.
#
#     Parameters
#     ----------
#     array
#         2D array of gridded data
#     transform
#         affine matrix transform
#     nodata
#         value representing no data in the given data
#     axis
#         `pyplot` axis to plot to
#     show
#         whether to show the plot
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
#     bounds = bounds_from_opposite_corners(origin, origin + numpy.flip(
#         array.shape) * resolution)
#     axis.matshow(array, extent=bounds[[0, 2, 1, 3]], aspect='auto', **kwargs)
#
#     if show:
#         pyplot.show()
#
#
# def plot_dataset(dataset: gdal.Dataset, index: int = 0,
#                  axis: pyplot.Axes = None, show: bool = False, **kwargs):
#     """
#     Plot the given GDAL dataset.
#
#     Parameters
#     ----------
#     dataset
#         GDAL dataset (raster or point cloud)
#     index
#         zero-based index of raster band / vector layer to read
#     axis
#         `pyplot` axis to plot to
#     show
#         whether to show the plot
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
#         plot_geoarray(raster_data.astype('float64'), transform, nodata, axis,
#                       show, **kwargs)
#     else:
#         plot_points(gdal_to_xyz(dataset), index, axis, show, **kwargs)
#
#
# def plot_interpolation(original_dataset: gdal.Dataset,
#                        interpolated_raster: gdal.Dataset,
#                        method: str, input_index: int = 0,
#                        output_index: int = 0,
#                        show: bool = False):
#     """
#     Plot original data side-by-side with an interpolated raster for
#     comparison.
#
#     Parameters
#     ----------
#     original_dataset
#         GDAL dataset (point cloud or raster) of original data
#     interpolated_raster
#         GDAL raster of interpolated data
#     method
#         method of interpolation
#     input_index
#         zero-based index of layer / band to read from the input dataset
#     output_index
#         zero-based index of band to read from the output raster
#     show
#         whether to show the plot
#     """
#
#     if original_dataset.RasterCount > 0:
#         original_raster_band = original_dataset.GetRasterBand(input_index
#         + 1)
#         original_data = original_raster_band.ReadAsArray()
#         original_nodata = original_raster_band.GetNoDataValue()
#         original_data[original_data == original_nodata] = numpy.nan
#     else:
#         original_data = gdal_to_xyz(original_dataset)[:, input_index + 2]
#
#     interpolated_raster_band = interpolated_raster.GetRasterBand(
#         output_index + 1)
#     interpolated_nodata = interpolated_raster_band.GetNoDataValue()
#     interpolated_data = interpolated_raster_band.ReadAsArray()
#     interpolated_data[interpolated_data == interpolated_nodata] = numpy.nan
#
#     # get minimum and maximum values for all three dimensions
#     z_values = numpy.concatenate(
#         (numpy.ravel(interpolated_data), numpy.ravel(original_data)))
#     min_z = numpy.nanmin(z_values)
#     max_z = numpy.nanmax(z_values)
#
#     # create a new figure window with two subplots
#     figure = pyplot.figure()
#     left_axis = figure.add_subplot(1, 2, 1)
#     left_axis.set_title('survey data')
#     right_axis = figure.add_subplot(1, 2, 2, sharex=left_axis,
#                                     sharey=left_axis)
#     right_axis.set_title(f'{method} interpolation to raster')
#
#     # plot data
#     plot_dataset(original_dataset, input_index, left_axis, vmin=min_z,
#                  vmax=max_z)
#     plot_dataset(interpolated_raster, input_index, right_axis, vmin=min_z,
#                  vmax=max_z)
#     right_axis.axes.get_yaxis().set_visible(False)
#
#     # create colorbar
#     figure.colorbar(ScalarMappable(norm=Normalize(vmin=min_z, vmax=max_z)),
#                     ax=(right_axis, left_axis))
#
#     # pause program execution and show the figure
#     if show:
#         pyplot.show()
#
#
# def _maxValue(arr: numpy.array):
#     """
#     Returns the most used value in the array as an integer
#
#     Takes an input array and finds the most used value in the array, this
#     value is used by the program to assume the array's nodata value
#
#     Parameters
#     ----------
#     arr: numpy.array :
#         An input array
#
#     Returns
#     -------
#
#     """
#
#     nums, counts = numpy.unique(arr, return_counts=True)
#     index = numpy.where(counts == numpy.amax(counts))
#     return int(nums[index])


def download_coastline(overwrite: bool = False) -> pathlib.Path:
    data_directory = pathlib.Path(appdirs.user_data_dir('ne_coastline'))
    if not data_directory.exists():
        data_directory.mkdir(exist_ok=True, parents=True)

    coastline_filename = data_directory / 'ne_110m_coastline.shp'

    if not coastline_filename.exists() or overwrite:
        # download and save if not present
        url = 'http://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip'
        response = requests.get(url, stream=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            for member_filename in zip_file.namelist():
                file_data = zip_file.read(member_filename)
                with open(data_directory / member_filename, 'wb') as output_file:
                    output_file.write(file_data)
        assert coastline_filename.exists(), 'coastline file not downloaded'

    return coastline_filename


def plot_coastline(axis: Axes = None, show: bool = False, save_filename: PathLike = None):
    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

    coastline_filename = download_coastline()
    dataframe = geopandas.read_file(coastline_filename)
    dataframe.plot(ax=axis)

    if save_filename is not None:
        pyplot.savefig(save_filename)

    if show:
        pyplot.show()
