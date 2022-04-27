import math
from os import PathLike
from pathlib import Path
from typing import Dict, List

import cartopy
import geopandas
from matplotlib import gridspec, pyplot
from matplotlib.axis import Axis
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy
import xarray

from ensembleperturbation.plotting.geometry import plot_points, plot_surface
from ensembleperturbation.plotting.nodes import plot_node_map
from ensembleperturbation.plotting.utilities import colorbar_axis


def get_validation_statistics(O, P, decimals):
    # pearson correlation coefficient
    def corrcoef(O, P):
        MP = P.mean()
        MO = O.mean()
        PD2 = ((P - MP) ** 2).sum()
        OD2 = ((O - MO) ** 2).sum()
        PDOD = ((P - MP) * (O - MO)).sum()
        corr = PDOD / (PD2 * OD2) ** 0.5
        if numpy.isnan(corr.values):
            corr.values = 1.0  # 0/0: perfect correlation
        return corr

    # correlation coefficient
    corr = corrcoef(O, P)
    # wet-dry correlation coefficient
    OW = ~numpy.isnan(O)
    PW = ~numpy.isnan(P)
    corr_wd = corrcoef(OW, PW)
    # mean bias
    mb = (P - O).mean()
    # mean absolute error
    mae = (abs(P - O)).mean()
    # root-mean-square error
    rmse = (((P - O) ** 2).mean()) ** 0.5

    return (
        numpy.round(corr.values, decimals),
        numpy.round(mb.values, decimals),
        numpy.round(mae.values, decimals),
        numpy.round(rmse.values, decimals),
        numpy.round(corr_wd.values, decimals),
    )


def comparison_plot_grid(
    variables: List[str], figure: Figure = None
) -> (Dict[str, Dict[str, Axis]], gridspec.GridSpec):
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


def plot_scatter_comparison(
    nodes: xarray.DataArray,
    title: str = None,
    output_filename: PathLike = None,
    reference_line: bool = True,
    statistics_text_offset: int = 0,
    figure: Figure = None,
    axes: Dict[str, Dict[str, Axis]] = None,
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
        figure.set_size_inches(10, 10 / 1.61803398875)

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
                min_value = numpy.nanmax([x.min(), y.min()])
                max_value = numpy.nanmin([x.max(), y.max()])
                axis.plot([min_value, max_value], [min_value, max_value], '--k', alpha=0.3)

            xlim = axis.get_xlim()
            ylim = axis.get_ylim()

            lim_min = min(xlim[0], ylim[0])
            lim_max = max(xlim[1], ylim[1])
            axis.set_xlim(lim_min, lim_max)
            axis.set_ylim(lim_min, lim_max)

            if statistics_text_offset > 0:
                ratio = statistics_text_offset * 0.1
                xlim = axis.get_xlim()
                ylim = axis.get_ylim()
                xpos = xlim[0] + ratio * (xlim[-1] - xlim[0])
                ypos = ylim[0] + numpy.array([0.95, 0.90, 0.85, 0.80]) * (ylim[-1] - ylim[0])
                corr, mb, mae, rmse, _ = get_validation_statistics(x, y, decimals=3)
                color = kwargs['c']
                axis.text(xpos, ypos[0], 'CORR = ' + str(corr), color=color)
                axis.text(xpos, ypos[1], 'MB = ' + str(mb) + ' m', color=color)
                axis.text(xpos, ypos[2], 'MAE = ' + str(mae) + ' m', color=color)
                axis.text(xpos, ypos[3], 'RMSE = ' + str(rmse) + ' m', color=color)

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')

    return figure, axes


def plot_boxplot_comparison(
    nodes: xarray.DataArray,
    title: str = None,
    output_filename: PathLike = None,
    subplot_integer: int = 111,
    figure: Figure = None,
    **kwargs,
):
    if 'source' not in nodes.dims:
        raise ValueError(f'"source" not found in data array dimensions: {nodes.dims}')
    elif len(nodes['source']) < 2:
        raise ValueError(f'cannot perform comparison with {len(nodes["source"])} source(s)')

    sources = nodes['source'].values
    dataset_type = nodes['type'].values

    if figure is None:
        figure = pyplot.figure()
        figure.set_size_inches(11, 11 / 1.61803398875)

    if title is not None:
        figure.suptitle(title)

    axis = figure.add_subplot(subplot_integer)

    num_runs = len(nodes['run'])
    rmse = numpy.empty(num_runs)
    corr = numpy.empty(num_runs)
    mb = numpy.empty(num_runs)
    mae = numpy.empty(num_runs)
    crwd = numpy.empty(num_runs)
    r = 0
    for run in nodes['run'].values:
        if not numpy.isnan(nodes.sel(run=run)).all():
            corr[r], mb[r], mae[r], rmse[r], crwd[r] = get_validation_statistics(
                nodes.sel(source='model', run=run),
                nodes.sel(source='surrogate', run=run),
                decimals=9,
            )
            r += 1

    c = 'gray'
    c2 = 'black'
    colors = ['blue', 'red']
    box = axis.boxplot(
        [corr[0:r], crwd[0:r], mb[0:r], mae[0:r], rmse[0:r]],
        whis=[5, 95],
        showfliers=True,
        patch_artist=True,
        widths=0.08,
        positions=[0.05, 0.2, 0.35, 0.5, 0.65],
        boxprops=dict(facecolor=c, color=c2, linewidth=0.25),
        whiskerprops=dict(color=c2, linewidth=1, linestyle='dashed'),
        medianprops=dict(color=c2, linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='x', markerfacecolor=c2, markersize=2),
    )
    for cdx, cap in enumerate(box['caps']):
        cap.set(color=colors[numpy.mod(cdx, 2)])
    for median in box['medians']:
        data = median.get_data()
        axis.text(
            data[0][1], data[1][0], str(numpy.round(data[1][0], 2)), ha='left', va='center'
        )

    pyplot.grid(True)
    pyplot.ylim([-1, 1])
    pyplot.xlim([-0.01, 0.75])
    axis.title.set_text(f'{dataset_type}: {r} runs')
    axis.set_xticklabels(['CORR', 'CORR$_{w/d}$', 'MB [m]', 'MAE [m]', 'RMSE [m]'])

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')

    return figure


def plot_sensitivities(
    sensitivities: xarray.Dataset, storm: str = None, output_filename: PathLike = None
):
    figure = pyplot.figure()
    figure.set_size_inches(12, 12 / 1.61803398875)
    figure.suptitle(
        f'Sobol sensitivities of {len(sensitivities["variable"])} variable(s) and {len(sensitivities["order"])} order(s) along {len(sensitivities["node"])} node(s)'
    )

    grid = gridspec.GridSpec(
        len(sensitivities['order']),
        len(sensitivities['variable']),
        figure=figure,
        wspace=0,
        hspace=0,
    )
    map_crs = cartopy.crs.PlateCarree()

    for order_index, order in enumerate(sensitivities['order']):
        for variable_index, variable in enumerate(sensitivities['variable']):
            axis = figure.add_subplot(
                grid[order_index, variable_index]
            )  # , projection=map_crs)
            order_variable_sensitivities = sensitivities.sel(order=order, variable=variable)

            plot_node_map(
                order_variable_sensitivities,
                map_title=None,
                colors=order_variable_sensitivities['sensitivities']
                if 'element' not in sensitivities
                else None,
                storm=storm,
                map_axis=axis,
                min_value=0,
                max_value=1,
            )

            if variable_index == 0:
                axis.yaxis.set_visible(True)
                axis.set_ylabel(str(order.values))
            elif variable_index > 0:
                axis.yaxis.set_visible(False)

            if order_index == 0:
                axis.xaxis.set_visible(True)
                axis.set_xlabel(str(variable.values))
                axis.xaxis.set_label_position('top')
                axis.xaxis.tick_top()
            elif order_index > 0:
                axis.xaxis.set_visible(False)

    colorbar_axis(
        normalization=Normalize(vmin=0, vmax=1),
        axis=pyplot.axes([-1, 0.1, 3, 0.15]),
        orientation='horizontal',
        own_axis=True,
        color_map='plasma',
    )

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def plot_validations(validation: xarray.Dataset, output_directory: PathLike):
    if output_directory is not None:
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)

    validation = validation['results']
    sources = validation['source'].values

    type_colors = {'training': 'b', 'validation': 'r'}
    fig_scatter = None
    fig_boxplot = None
    axes = None
    ncols = min(len(validation['type']), 2)
    subplot_integer = math.ceil(len(validation['type']) / ncols) * 100 + ncols * 10 + 1
    for index, result_type in enumerate(validation['type'].values):
        result_validation = validation.sel(type=result_type)
        fig_scatter, axes = plot_scatter_comparison(
            result_validation,
            title=f'comparison of {len(sources)} sources along {len(result_validation["node"])} node(s)',
            reference_line=index == 0,
            statistics_text_offset=2 * (index + 1),
            figure=fig_scatter,
            axes=axes,
            s=1,
            c=type_colors[result_type],
            label=result_type,
        )

        fig_boxplot = plot_boxplot_comparison(
            result_validation,
            title=f'error statistics of surrogate model',
            subplot_integer=subplot_integer + index,
            figure=fig_boxplot,
        )

    for row in axes.values():
        for axis in row.values():
            axis.legend()

    if output_directory is not None:
        fig_scatter.savefig(
            output_directory / f'validation_scatter.png', dpi=200, bbox_inches='tight',
        )
        fig_boxplot.savefig(
            output_directory / f'validation_boxplot.png', dpi=200, bbox_inches='tight',
        )


def plot_selected_validations(
    validation: xarray.Dataset, run_list: list, output_directory: PathLike
):
    sources = validation['source'].values
    if output_directory is not None:
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)

    bounds = numpy.array(
        [
            validation['x'].min(),
            validation['y'].min(),
            validation['x'].max(),
            validation['y'].max(),
        ]
    )
    vmax = numpy.round_(validation.sel(source='model').results.quantile(0.98), decimals=1)
    vmin = 0.0
    for run in run_list:
        figure = pyplot.figure()
        figure.set_size_inches(10, 10 / 1.61803398875)
        figure.suptitle(f'validation of surrogate model for run: {run}')

        for index, source in enumerate(sources):
            map_axis = figure.add_subplot(2, len(sources), index + 1)
            map_axis.title.set_text(f'{source}')
            countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

            map_axis.set_xlim((bounds[0], bounds[2]))
            map_axis.set_ylim((bounds[1], bounds[3]))

            xlim = map_axis.get_xlim()
            ylim = map_axis.get_ylim()

            countries.plot(color='lightgrey', ax=map_axis)

            points = numpy.vstack(
                (
                    validation['x'],
                    validation['y'],
                    validation.sel(type='validation', run=run, source=source).results,
                )
            ).T
            if 'element' not in validation:
                im = plot_points(
                    points=points, axis=map_axis, add_colorbar=False, vmax=vmax, vmin=vmin,
                )
            else:
                im = plot_surface(
                    points=points,
                    element_table=validation['element'].values,
                    axis=map_axis,
                    add_colorbar=False,
                    levels=numpy.linspace(vmin, vmax, 25 + 1),
                    extend='both',
                )

            map_axis.set_xlim(xlim)
            map_axis.set_ylim(ylim)

        pyplot.subplots_adjust(wspace=0.02, right=0.96)
        cax = pyplot.axes([0.95, 0.55, 0.015, 0.3])
        cbar = figure.colorbar(im, extend='both', cax=cax)

        if output_directory is not None:
            figure.savefig(
                output_directory / f'validation_{run}.png', dpi=200, bbox_inches='tight',
            )


def plot_selected_percentiles(
    node_percentiles: xarray.Dataset, perc_list: list, output_directory: PathLike
):
    percentiles = node_percentiles.quantiles

    sources = node_percentiles['source'].values
    if output_directory is not None:
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)

    bounds = numpy.array(
        [
            node_percentiles['x'].min(),
            node_percentiles['y'].min(),
            node_percentiles['x'].max(),
            node_percentiles['y'].max(),
        ]
    )
    vmax = numpy.round_(percentiles.sel(source='model').quantile(0.98), decimals=1)
    vmin = 0.0
    for perc in perc_list:
        figure = pyplot.figure()
        figure.set_size_inches(10, 10 / 1.61803398875)
        figure.suptitle(f'comparison of percentiles: {perc}%')
        for index, source in enumerate(sources):
            map_axis = figure.add_subplot(2, len(sources), index + 1)
            map_axis.title.set_text(f'{source}')
            countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

            map_axis.set_xlim((bounds[0], bounds[2]))
            map_axis.set_ylim((bounds[1], bounds[3]))

            xlim = map_axis.get_xlim()
            ylim = map_axis.get_ylim()

            countries.plot(color='lightgrey', ax=map_axis)

            points = numpy.vstack(
                (
                    node_percentiles['x'],
                    node_percentiles['y'],
                    percentiles.sel(quantile=perc, source=source),
                )
            ).T
            if 'element' not in node_percentiles:
                im = plot_points(
                    points=points, axis=map_axis, add_colorbar=False, vmax=vmax, vmin=vmin,
                )
            else:
                im = plot_surface(
                    points=points,
                    element_table=node_percentiles['element'].values,
                    axis=map_axis,
                    add_colorbar=False,
                    levels=numpy.linspace(vmin, vmax, 25 + 1),
                    extend='both',
                )

            map_axis.set_xlim(xlim)
            map_axis.set_ylim(ylim)

        pyplot.subplots_adjust(wspace=0.02, right=0.96)
        cax = pyplot.axes([0.95, 0.55, 0.015, 0.3])
        cbar = figure.colorbar(im, extend='both', cax=cax)

        if output_directory is not None:
            figure.savefig(
                output_directory / f'percentiles_{perc}.png', dpi=200, bbox_inches='tight',
            )


def plot_kl_surrogate_fit(
    kl_fit: xarray.Dataset,
    output_filename: PathLike,
    reference_line: bool = True,
    statistics_text: bool = True,
):
    kl_fit = kl_fit['results']

    alim = [kl_fit.min(), kl_fit.max()]
    subplot_width = 3
    subplot_height = numpy.ceil(len(kl_fit['node']) / subplot_width).astype(int)

    figure = pyplot.figure()
    figure.set_size_inches(11, 11 * subplot_height / 5 / 1.61803398875)
    figure.suptitle(f'comparison of surrogate for the KL samples')

    for mode in range(len(kl_fit['node'])):
        axis = figure.add_subplot(subplot_height, subplot_width, mode + 1)
        qoi = kl_fit.sel(node=mode, source='model')
        qoi_pc = kl_fit.sel(node=mode, source='surrogate')

        axis.plot(qoi, qoi_pc, 'o', markersize=4)

        if reference_line:
            axis.plot([alim[0], alim[-1]], [alim[0], alim[-1]], '--k', alpha=0.3)

        if statistics_text:
            xpos = alim[0] + 0.1 * (alim[-1] - alim[0])
            ypos = alim[0].values + numpy.array([0.85, 0.7]) * (alim[-1] - alim[0]).values
            corr, _, _, rmse, _ = get_validation_statistics(qoi, qoi_pc, decimals=3)
            axis.text(xpos, ypos[0], 'RMSE = ' + str(rmse))
            axis.text(xpos, ypos[1], 'CORR = ' + str(corr))

        axis.set_xlim(alim)
        axis.set_ylim(alim)
        if mode + 1 > (subplot_height - 1) * subplot_width:
            axis.set_xlabel('actual')
        axis.set_ylabel('predicted')
        axis.title.set_text(f'KL mode-{mode + 1}')

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')
