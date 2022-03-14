from os import PathLike

import cartopy
import geopandas
from matplotlib import cm, gridspec, pyplot
from matplotlib.axis import Axis
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.tri import Triangulation
import numpy
from stormevents.nhc import VortexTrack
import xarray

from ensembleperturbation.plotting.utilities import colorbar_axis


def node_color_map(
    nodes: xarray.Dataset,
    colors: list = None,
    min_value: float = None,
    max_value: float = None,
    logarithmic: bool = False,
) -> (numpy.ndarray, Normalize, Colormap, numpy.ndarray):
    if colors is None:
        color_map = cm.get_cmap('plasma')
        color_values = numpy.arange(len(nodes['node']))
        normalization = Normalize(vmin=numpy.min(color_values), vmax=numpy.max(color_values))
        colors = color_map(normalization(color_values))
    elif isinstance(colors, str):
        color_map = cm.get_cmap('plasma')
        color_values = nodes[colors]
        if len(color_values.dims) > 1:
            color_values = color_values.mean(
                [dim for dim in color_values.dims if dim != 'node']
            )
        if min_value is None:
            min_value = float(color_values.min().values)
        if max_value is None:
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

        color_map = cm.get_cmap('plasma')
        if min_value is None:
            min_value = numpy.nanmin(colors)
        if max_value is None:
            max_value = numpy.nanmax(colors)

        if logarithmic:
            normalization = LogNorm(vmin=min_value, vmax=max_value)
        else:
            normalization = Normalize(vmin=min_value, vmax=max_value)

        if (len(colors.shape) < 2 or colors.shape[1] != 4) and numpy.any(~numpy.isnan(colors)):
            color_values = colors
            colors = color_map(normalization(color_values))
        else:
            color_values = None

    return color_values, normalization, color_map, colors


def plot_nodes_across_runs(
    nodes: xarray.Dataset,
    title: str = None,
    colors: [] = None,
    storm: str = None,
    output_filename: PathLike = None,
    min_value: float = None,
    max_value: float = None,
    logarithmic: bool = False,
):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    if title is not None:
        figure.suptitle(title)

    grid = gridspec.GridSpec(len(nodes.data_vars), 2, figure=figure)

    map_crs = cartopy.crs.PlateCarree()
    map_axis = figure.add_subplot(grid[:, 0], projection=map_crs)

    color_values, normalization, color_map, map_colors = node_color_map(
        nodes, colors=colors, min_value=min_value, max_value=max_value, logarithmic=logarithmic
    )
    plot_node_map(
        nodes, colors=map_colors, storm=storm, map_axis=map_axis, logarithmic=logarithmic
    )

    if colors is not None:
        colorbar_axis(
            normalization=Normalize(
                vmin=numpy.nanmin(color_values), vmax=numpy.nanmax(color_values)
            ),
            axis=map_axis,
            orientation='vertical',
        )

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


def plot_node_map(
    nodes: xarray.Dataset,
    map_title: str = None,
    colors: list = None,
    storm: str = None,
    map_axis: Axis = None,
    min_value: float = None,
    max_value: float = None,
    logarithmic: bool = False,
):
    if isinstance(colors, str) and map_title is not None:
        map_title = f'"{colors}" of {map_title}'

    data_var_name = [i for i in nodes.data_vars]
    color_values, normalization, color_map, colors = node_color_map(
        nodes,
        colors=colors,
        min_value=min_value,
        max_value=max_value,
        logarithmic=logarithmic,
    )

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
    countries.plot(color='lightgrey', ax=map_axis)

    if storm is not None:
        if not isinstance(storm, VortexTrack):
            try:
                storm = VortexTrack.from_fort22(storm)
            except FileNotFoundError:
                storm = VortexTrack(storm)

        map_axis.plot(
            storm.data['longitude'], storm.data['latitude'], label=storm.name,
        )

        if storm.name is not None:
            map_axis.legend(fontsize=6)

    if 'element' in nodes:
        mesh_tri = Triangulation(
            nodes.coords['x'],
            nodes.coords['y'],
            triangles=nodes.coords['element'],
            mask=numpy.isnan(nodes[data_var_name[0]][nodes.coords['element']]).any(axis=1),
        )
        levels = numpy.linspace(min_value, max_value, 26)
        map_axis.tricontourf(
            mesh_tri,
            nodes[data_var_name[0]].values,
            levels=levels,
            cmap=color_map,  # transform=map_crs,
        )
    else:
        map_axis.scatter(
            x=nodes.coords['x'],
            y=nodes.coords['y'],
            c=colors,
            s=2,
            norm=normalization,  # transform=map_crs,
        )

    map_axis.set_xlim(map_bounds[0], map_bounds[2])
    map_axis.set_ylim(map_bounds[1], map_bounds[3])

    if map_title is not None:
        map_axis.set_title(map_title)
