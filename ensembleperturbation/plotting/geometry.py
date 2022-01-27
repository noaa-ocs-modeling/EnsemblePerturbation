from os import PathLike
from typing import List, Union

from matplotlib import pyplot
from matplotlib.axis import Axis
from matplotlib.cm import get_cmap
import numpy
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape


def plot_polygon(
    geometry: Union[Polygon, MultiPolygon], axis: Axis = None, show: bool = False, **kwargs,
):
    """
    plot the given polygon

    :param geometry: Shapely polygon (or multipolygon)
    :param axis: `pyplot` axis to plot to
    :param show: whether to show the plot
    """

    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

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
    geometries: List[Polygon],
    colors: List[str] = None,
    axis: Axis = None,
    show: bool = False,
    **kwargs,
):
    """
    plot the given polygons using the given colors

    :param geometries: list of shapely polygons or multipolygons
    :param colors: colors to plot each region
    :param axis: `pyplot` axis to plot to
    :param show: whether to show the plot
    """

    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

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
    sw: (float, float), ne: (float, float), axis: Axis = None, show: bool = False, **kwargs,
):
    """
    plot the bounding box of the given extent

    :param sw: XY coordinates of southwest corner
    :param ne: XY coordinates of northeast corner
    :param axis: `pyplot` axis to plot to
    :param show: whether to show the plot
    """

    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

    corner_points = numpy.array([sw, (ne[0], sw[1]), ne, (sw[0], ne[1]), sw])

    axis.plot(corner_points[:, 0], corner_points[:, 1], **kwargs)

    if show:
        pyplot.show()


def plot_points(
    points: Union[numpy.ndarray, MultiPoint],
    index: int = 0,
    axis: Axis = None,
    show: bool = False,
    save_filename: PathLike = None,
    title: str = None,
    add_colorbar: bool = True,
    **kwargs,
):
    """
    create a scatter plot of the given points

    :param points: N x M array of points
    :param index: zero-based index of vector layer to read
    :param axis: `pyplot` axis to plot to
    :param show: whether to show the plot
    :param save_filename: whether to save the plot
    :param title: whether to add a title to the plot
    """

    if type(points) is MultiPoint:
        points = numpy.squeeze(numpy.stack((point._get_coords() for point in points), axis=0))

    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

    if 'c' not in kwargs and points.shape[1] > 2:
        kwargs['c'] = points[:, index + 2]

    if 's' not in kwargs:
        kwargs['s'] = 2

    sc = axis.scatter(points[:, 0], points[:, 1], **kwargs)

    if 'c' in kwargs and add_colorbar:
        pyplot.colorbar(sc)

    if title is not None:
        pyplot.title(title)

    if save_filename is not None:
        pyplot.savefig(save_filename)

    if show:
        pyplot.show()
    else:
        pyplot.close()

    return sc
