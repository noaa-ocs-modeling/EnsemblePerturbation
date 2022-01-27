from matplotlib import cm, pyplot
from matplotlib.axis import Axis
from matplotlib.colors import Normalize


def colorbar_axis(
    normalization: Normalize,
    axis: Axis = None,
    color_map: str = None,
    orientation: str = None,
    own_axis: bool = False,
) -> Axis:
    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

    if color_map is None:
        color_map = 'jet'
    color_map = cm.get_cmap(color_map)

    if orientation is None:
        orientation = 'horizontal'

    if own_axis:
        axis.set_visible(False)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    return axis.figure.colorbar(
        mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
        orientation=orientation,
        ax=axis,
    )
