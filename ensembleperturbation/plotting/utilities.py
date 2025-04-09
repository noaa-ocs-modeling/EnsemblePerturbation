from matplotlib import cm, pyplot
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.colors import Normalize


def colorbar_axis(
    normalization: Normalize,
    figure: Figure = None,
    axis: Axis = None,
    color_map: str = None,
    orientation: str = None,
    own_axis: bool = False,
    extend: str = None,
    shrink: float = None,
    label: str = None,
) -> Axis:
    if figure is None:
        figure = pyplot.figure()
    if axis is None:
        axis = figure.add_subplot(1, 1, 1)

    if color_map is None:
        color_map = 'viridis'
    color_map = cm.get_cmap(color_map)

    if orientation is None:
        orientation = 'horizontal'

    if own_axis:
        axis.set_visible(False)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    return figure.colorbar(
        mappable=cm.ScalarMappable(cmap=color_map, norm=normalization),
        orientation=orientation,
        ax=axis,
        extend=extend,
        shrink=shrink,
        label=label,
    )
