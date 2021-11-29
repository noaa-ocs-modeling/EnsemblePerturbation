import io
import math
from os import PathLike
import pathlib
from pathlib import Path
from typing import Union
import zipfile

# from affine import Affine
from adcircpy.forcing import BestTrackForcing
import appdirs

# import gdal
import cartopy
import geopandas
from matplotlib import cm, gridspec, pyplot
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from modelforcings.vortex import VortexForcing
import numpy
import requests
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape
import xarray

from ensembleperturbation.utilities import encode_categorical_values, get_logger

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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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
    save_filename: str = None,
    title: str = None,
    **kwargs,
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
    save_filename
        whether to save the plot
    title
        whether to add a title to the plot
    """

    if type(points) is MultiPoint:
        points = numpy.squeeze(numpy.stack((point._get_coords() for point in points), axis=0))

    if axis is None:
        axis = pyplot.gca()

    if 'c' not in kwargs and points.shape[1] > 2:
        kwargs['c'] = points[:, index + 2]

    if 's' not in kwargs:
        kwargs['s'] = 2

    sc = axis.scatter(points[:, 0], points[:, 1], **kwargs)

    if 'c' in kwargs:
        pyplot.colorbar(sc)

    if title is not None:
        pyplot.title(title)

    if save_filename is not None:
        pyplot.savefig(save_filename)

    if show:
        pyplot.show()
    else:
        pyplot.close()


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


def node_color_map(
    nodes: xarray.Dataset, colors: [] = None
) -> (numpy.ndarray, Normalize, Colormap, numpy.ndarray):
    if colors is None:
        color_map = cm.get_cmap('jet')
        color_values = numpy.arange(len(nodes['node']))
        normalization = Normalize(vmin=numpy.min(color_values), vmax=numpy.max(color_values))
        colors = color_map(normalization(color_values))
    elif isinstance(colors, str):
        color_map = cm.get_cmap('jet')
        color_values = nodes[colors]
        if len(color_values.dims) > 1:
            color_values = color_values.mean(
                [dim for dim in color_values.dims if dim != 'node']
            )
        min_value = float(color_values.min().values)
        max_value = float(color_values.max().values)
        try:
            normalization = LogNorm(vmin=min_value, vmax=max_value)
            normalized_color_values = normalization(color_values)
        except ValueError:
            normalization = Normalize(vmin=min_value, vmax=max_value)
            normalized_color_values = normalization(color_values)
        colors = color_map(normalized_color_values)
    else:
        colors = numpy.array(colors)

        color_map = cm.get_cmap('jet')
        min_value = numpy.min(colors)
        max_value = numpy.max(colors)
        try:
            normalization = LogNorm(vmin=min_value, vmax=max_value)
        except ValueError:
            normalization = Normalize(vmin=min_value, vmax=max_value)

        if len(colors.shape) < 2 or colors.shape[1] != 4:
            color_values = colors
            colors = color_map(normalization(color_values))
        else:
            color_values = None

    return color_values, normalization, color_map, colors


def plot_node_map(
    nodes: xarray.Dataset,
    map_title: str = None,
    colors: [] = None,
    storm: str = None,
    map_axis: pyplot.Axes = None,
):
    if map_title is None:
        map_title = f'{len(nodes["node"])} nodes'
    if isinstance(colors, str):
        map_title = f'"{colors}" of {map_title}'

    color_values, normalization, color_map, colors = node_color_map(nodes, colors=colors)

    map_crs = cartopy.crs.PlateCarree()
    if map_axis is None:
        map_axis = pyplot.subplot(1, 1, 1, projection=map_crs)

    map_bounds = [
        float(nodes.coords['x'].min().values),
        float(nodes.coords['y'].min().values),
        float(nodes.coords['x'].max().values),
        float(nodes.coords['y'].max().values),
    ]

    countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    if storm is not None and not isinstance(storm, VortexForcing):
        try:
            storm = BestTrackForcing.from_fort22(storm)
        except FileNotFoundError:
            storm = BestTrackForcing(storm)

    countries.plot(color='lightgrey', ax=map_axis)
    storm.data.plot(
        x='longitude',
        y='latitude',
        ax=map_axis,
        label=storm.name,
        legend=storm.name is not None,
    )

    map_axis.scatter(
        x=nodes['x'], y=nodes['y'], c=colors, s=2, norm=normalization, transform=map_crs
    )

    map_axis.set_xlim(map_bounds[0], map_bounds[2])
    map_axis.set_ylim(map_bounds[1], map_bounds[3])
    map_axis.set_title(map_title)


def plot_nodes_across_runs(
    nodes: xarray.Dataset,
    title: str = None,
    colors: [] = None,
    storm: str = None,
    output_filename: PathLike = None,
):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    if title is not None:
        figure.suptitle(title)

    grid = gridspec.GridSpec(len(nodes.data_vars), 2, figure=figure)

    map_crs = cartopy.crs.PlateCarree()
    map_axis = figure.add_subplot(grid[:, 0], projection=map_crs)

    color_values, normalization, color_map, map_colors = node_color_map(nodes, colors=colors)
    plot_node_map(nodes, colors=map_colors, storm=storm, map_axis=map_axis)

    shared_axis = None
    for variable_index, (variable_name, variable) in enumerate(nodes.data_vars.items()):
        axis_kwargs = {}
        if shared_axis is not None:
            axis_kwargs['sharex'] = shared_axis

        variable_axis = figure.add_subplot(grid[variable_index, 1], **axis_kwargs)

        if shared_axis is None:
            shared_axis = variable_axis

        if variable_index < len(nodes.data_vars) - 1:
            variable_axis.get_xaxis().set_visible(False)

        if 'source' in variable.dims:
            sources = variable['source']
        else:
            sources = [None]

        for source_index, source in enumerate(sources):
            kwargs = {}
            if source == 'model':
                variable_colors = cm.get_cmap('jet')(color_values)
            elif source == 'surrogate':
                variable_colors = 'grey'
                kwargs['linestyle'] = '--'
            else:
                variable_colors = cm.get_cmap('jet')(color_values)

            if 'source' in variable.dims:
                source_data = variable.sel(source=source)
            else:
                source_data = variable

            if 'time' in nodes.dims:
                for node_index in range(len(nodes['node'])):
                    node_data = source_data.isel(node=node_index)
                    node_color = variable_colors[node_index]
                    node_data.plot.line(
                        x='time', c=node_color, ax=variable_axis, **kwargs,
                    )
                    if variable_name == 'mean' and 'std' in nodes.data_vars:
                        std_data = nodes['std'].isel(node=node_index)
                        if 'source' in std_data.dims:
                            std_data = std_data.sel(source=source)
                        variable_axis.fill_between(
                            nodes['time'],
                            node_data - std_data,
                            node_data + std_data,
                            color=node_color,
                            alpha=0.3,
                            **kwargs,
                        )
            else:
                bar_width = 1
                bar_offset = bar_width * (source_index + 0.5 - len(sources) / 2)

                variable_axis.bar(
                    x=numpy.arange(len(source_data)) + bar_offset,
                    width=bar_width,
                    height=source_data,
                    color=variable_colors,
                    **kwargs,
                )
                variable_axis.set_ylim([min(0, source_data.min()), source_data.max()])

        variable_axis.set_title(variable_name)
        variable_axis.tick_params(axis='x', which='both', labelsize=6)
        variable_axis.set(xlabel=None)

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def comparison_plot_grid(
    variables: [str], figure: Figure = None
) -> ({str: {str: Axis}}, gridspec.GridSpec):
    if figure is None:
        figure = pyplot.figure()

    num_variables = len(variables)
    num_plots = int(num_variables * (num_variables - 1) / 2)

    grid_length = math.ceil(num_plots ** 0.5)
    grid = gridspec.GridSpec(grid_length, grid_length, figure=figure, wspace=0, hspace=0)

    shared_grid_columns = {'x': {}, 'y': {}}
    for plot_index in range(num_plots):
        original_index = plot_index
        if plot_index >= num_variables:
            # TODO fix this
            plot_index = (plot_index % num_variables) + 1 if original_index % 2 else 3

        variable = variables[plot_index]

        if original_index % 2 == 0:
            shared_grid_columns['x'][variable] = None
        else:
            shared_grid_columns['y'][variable] = None

    if len(shared_grid_columns['y']) == 0:
        shared_grid_columns['y'][variables[-1]] = None

    axes = {}
    for row_index in range(grid_length):
        row_variable = list(shared_grid_columns['y'])[row_index]
        axes[row_variable] = {}
        for column_index in range(grid_length - row_index):
            column_variable = list(shared_grid_columns['x'])[column_index]

            sharex = shared_grid_columns['x'][column_variable]
            sharey = shared_grid_columns['y'][row_variable]

            variable_axis = figure.add_subplot(
                grid[row_index, column_index], sharex=sharex, sharey=sharey,
            )

            if sharex is None:
                shared_grid_columns['x'][column_variable] = variable_axis
            if sharey is None:
                shared_grid_columns['y'][row_variable] = variable_axis

            if grid_length != 1:
                if row_index == 0:
                    variable_axis.set_xlabel(column_variable)
                    variable_axis.xaxis.set_label_position('top')
                    variable_axis.xaxis.tick_top()
                    if row_index == grid_length - column_index - 1:
                        variable_axis.secondary_xaxis('bottom')
                elif row_index != grid_length - column_index - 1:
                    variable_axis.xaxis.set_visible(False)

                if column_index == 0:
                    variable_axis.set_ylabel(row_variable)
                    if row_index == grid_length - column_index - 1:
                        variable_axis.secondary_yaxis('right')
                elif row_index == grid_length - column_index - 1:
                    variable_axis.yaxis.tick_right()
                else:
                    variable_axis.yaxis.set_visible(False)
            else:
                variable_axis.set_xlabel(column_variable)
                variable_axis.set_ylabel(row_variable)

            axes[row_variable][column_variable] = variable_axis

    return axes, grid


def plot_perturbed_variables(
    perturbations: xarray.Dataset, title: str = None, output_filename: PathLike = None,
):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    if title is None:
        title = f'{len(perturbations["run"])} pertubation(s) of {len(perturbations["variable"])} variable(s)'
    figure.suptitle(title)

    variables = perturbations['variable'].values
    axes, grid = comparison_plot_grid(variables, figure=figure)

    color_map = cm.get_cmap('jet')

    perturbation_colors = perturbations['weights']
    if not perturbation_colors.isnull().values.all():
        color_map_axis = figure.add_subplot(grid[-1, -1])
        color_map_axis.set_visible(False)
        min_value = float(perturbation_colors.min().values)
        max_value = float(perturbation_colors.max().values)
        perturbation_colors.loc[perturbation_colors.isnull()] = 0
        try:
            normalization = LogNorm(vmin=min_value, vmax=max_value)
            colorbar = figure.colorbar(
                mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
                orientation='horizontal',
                ax=color_map_axis,
            )
        except ValueError:
            normalization = Normalize(vmin=min_value, vmax=max_value)
            colorbar = figure.colorbar(
                mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
                orientation='horizontal',
                ax=color_map_axis,
            )
        colorbar.set_label('weight')
    else:
        perturbation_colors = numpy.arange(len(perturbation_colors))
        normalization = None

    perturbations = perturbations['perturbations']
    for row_variable, columns in axes.items():
        for column_variable, axis in columns.items():
            axis.scatter(
                perturbations.sel(variable=column_variable),
                perturbations.sel(variable=row_variable),
                c=perturbation_colors,
                cmap=color_map,
                norm=normalization,
            )

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def plot_comparison(
    nodes: xarray.DataArray,
    title: str = None,
    output_filename: PathLike = None,
    reference_line: bool = True,
    figure: Figure = None,
    axes: {str: {str: Axis}} = None,
    **kwargs,
):
    if 'source' not in nodes.dims:
        raise ValueError(f'"source" not found in data array dimensions: {nodes.dims}')
    elif len(nodes['source']) < 2:
        raise ValueError(f'cannot perform comparison with {len(nodes["source"])} source(s)')

    if 'c' not in kwargs and 'weights' in nodes:
        kwargs['c'] = nodes['weights']

    sources = nodes['source'].values

    if figure is None and axes is None:
        figure = pyplot.figure()
        figure.set_size_inches(12, 12 / 1.61803398875)

    if axes is None:
        axes, _ = comparison_plot_grid(sources, figure=figure)

    if title is not None:
        figure.suptitle(title)

    for row_source, columns in axes.items():
        for column_source, axis in columns.items():
            x = nodes.sel(source=column_source)
            y = nodes.sel(source=row_source)
            axis.scatter(x, y, **kwargs)

            if reference_line:
                xlim = axis.get_xlim()
                ylim = axis.get_ylim()

                min_value = numpy.nanmax([x.min(), y.min()])
                max_value = numpy.nanmin([x.max(), y.max()])
                axis.plot([min_value, max_value], [min_value, max_value], '--k', alpha=0.3)

                axis.set_xlim(xlim)
                axis.set_ylim(ylim)

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')

    return axes


def plot_perturbations(
    training_perturbations: xarray.Dataset,
    validation_perturbations: xarray.Dataset,
    runs: [str],
    perturbation_types: [str],
    track_directory: PathLike = None,
    output_directory: PathLike = None,
):
    if output_directory is not None:
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)

    plot_perturbed_variables(
        training_perturbations,
        title=f'{len(training_perturbations["run"])} training pertubation(s) of {len(training_perturbations["variable"])} variable(s)',
        output_filename=output_directory / 'training_perturbations.png'
        if output_directory is not None
        else None,
    )
    plot_perturbed_variables(
        validation_perturbations,
        title=f'{len(validation_perturbations["run"])} validation pertubation(s) of {len(validation_perturbations["variable"])} variable(s)',
        output_filename=output_directory / 'validation_perturbations.png'
        if output_directory is not None
        else None,
    )

    if track_directory is not None:
        if not isinstance(track_directory, Path):
            track_directory = Path(track_directory)

        if track_directory.exists():
            track_filenames = {
                track_filename.stem: track_filename
                for track_filename in track_directory.glob('*.22')
            }

            figure = pyplot.figure()
            figure.set_size_inches(12, 12 / 1.61803398875)
            figure.suptitle(f'{len(track_filenames)} perturbations of storm track')

            map_axis = figure.add_subplot(1, 1, 1)
            countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

            unique_perturbation_types = numpy.unique(perturbation_types)
            encoded_perturbation_types = encode_categorical_values(
                perturbation_types, unique_values=unique_perturbation_types
            )
            linear_normalization = Normalize()
            colors = get_cmap('jet')(linear_normalization(encoded_perturbation_types))

            bounds = numpy.array([None, None, None, None])
            for index, run in enumerate(runs):
                storm = VortexForcing.from_fort22(track_filenames[run]).data
                points = storm.loc[:, ['longitude', 'latitude']].values.reshape(-1, 1, 2)
                segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
                line_collection = LineCollection(
                    segments,
                    linewidths=numpy.concatenate(
                        [
                            [0],
                            storm['radius_of_maximum_winds']
                            / max(storm['radius_of_maximum_winds']),
                        ]
                    )
                    * 4,
                    color=colors[index],
                )
                map_axis.add_collection(line_collection)

                track_bounds = numpy.array(
                    [
                        points[:, :, 0].min(),
                        points[:, :, 1].min(),
                        points[:, :, 0].max(),
                        points[:, :, 1].max(),
                    ]
                )
                if bounds[0] is None or track_bounds[0] < bounds[0]:
                    bounds[0] = track_bounds[0]
                if bounds[1] is None or track_bounds[1] < bounds[1]:
                    bounds[1] = track_bounds[1]
                if bounds[2] is None or track_bounds[2] > bounds[2]:
                    bounds[2] = track_bounds[2]
                if bounds[3] is None or track_bounds[3] > bounds[3]:
                    bounds[3] = track_bounds[3]

            map_axis.set_xlim((bounds[0], bounds[2]))
            map_axis.set_ylim((bounds[1], bounds[3]))

            unique_perturbation_type_colors = get_cmap('jet')(
                linear_normalization(numpy.unique(encoded_perturbation_types))
            )
            map_axis.legend(
                [Patch(facecolor=color) for color in unique_perturbation_type_colors],
                unique_perturbation_types,
            )

            xlim = map_axis.get_xlim()
            ylim = map_axis.get_ylim()

            countries.plot(color='lightgrey', ax=map_axis)

            map_axis.set_xlim(xlim)
            map_axis.set_ylim(ylim)

            if output_directory is not None:
                figure.savefig(
                    output_directory / 'storm_tracks.png', dpi=200, bbox_inches='tight',
                )


def plot_sensitivities(sensitivities: xarray.Dataset, output_filename: PathLike = None):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    figure.suptitle(
        f'Sobol sensitivities of {len(sensitivities["variable"])} variables along {len(sensitivities["node"])} node(s)'
    )

    axis = figure.add_subplot(1, 1, 1)

    for variable in sensitivities['variable']:
        axis.scatter(
            sensitivities['node'], sensitivities.sel(variable=variable), label=variable
        )

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def plot_validations(validation: xarray.Dataset, output_filename: PathLike):
    validation = validation['results']

    sources = validation['source'].values

    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    figure.suptitle(
        f'comparison of {len(sources)} sources along {len(validation["node"])} node(s)'
    )

    type_colors = {'training': 'b', 'validation': 'r'}
    axes = None
    for index, result_type in enumerate(validation['type'].values):
        result_validation = validation.sel(type=result_type)
        axes = plot_comparison(
            result_validation,
            title=f'comparison of {len(sources)} sources along {len(result_validation["node"])} node(s)',
            reference_line=index == 0,
            figure=figure,
            axes=axes,
            s=1,
            c=type_colors[result_type],
            label=result_type,
        )

    for row in axes.values():
        for axis in row.values():
            axis.legend()

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')
