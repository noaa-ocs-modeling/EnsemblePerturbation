from os import PathLike
from pathlib import Path
from typing import List

import geopandas
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
import numpy
from stormevents.nhc import VortexTrack
import xarray

from ensembleperturbation.plotting.surrogate import comparison_plot_grid
from ensembleperturbation.plotting.utilities import colorbar_axis
from ensembleperturbation.utilities import encode_categorical_values


def plot_perturbed_variables(
    perturbations: xarray.Dataset, title: str = None, output_filename: PathLike = None,
):
    figure = pyplot.figure()
    figure.set_size_inches(10, 10 / 1.61803398875)
    if title is None:
        title = f'{len(perturbations["run"])} pertubation(s) of {len(perturbations["variable"])} variable(s)'
    figure.suptitle(title)

    variables = perturbations['variable'].values
    axes, grid = comparison_plot_grid(variables, figure=figure)

    perturbation_colors = perturbations['weights']
    if (
        perturbation_colors.isnull().values.all()
        or (perturbation_colors == perturbation_colors[0]).values.all()
    ):
        perturbation_colors = numpy.arange(len(perturbation_colors))
        perturbation_size = (6 * numpy.ones(len(perturbation_colors))) ** 2
        alpha = 0.9

        normalization = None
        color_map = get_cmap('nipy_spectral')
    else:
        min_value = float(perturbation_colors.min().values)
        max_value = float(perturbation_colors.max().values)

        color_map = get_cmap('Greys')
        orientation = 'horizontal'

        if min_value > 0:
            normalization = LogNorm(vmin=min_value, vmax=max_value)
        else:
            # normalization = SymLogNorm(0.01, vmin=min_value, vmax=max_value)
            normalization = Normalize(vmin=min_value, vmax=max_value)
        colorbar = colorbar_axis(
            normalization=normalization,
            axis=figure.add_subplot(grid[-1, -1]),
            orientation=orientation,
            color_map=color_map,
            own_axis=True,
        )
        colorbar.set_label('weight')

        perturbation_colors.loc[perturbation_colors.isnull()] = 0
        perturbation_size = (
            12 * (perturbation_colors.values - min_value) / (max_value - min_value)
        ) ** 2
        alpha = 0.75

    perturbations = perturbations['perturbations']
    sort_idx = numpy.argsort(perturbation_size)[::-1]
    for row_variable, columns in axes.items():
        for column_variable, axis in columns.items():
            axis.scatter(
                perturbations.sel(variable=column_variable)[sort_idx],
                perturbations.sel(variable=row_variable)[sort_idx],
                c=perturbation_colors[sort_idx],
                s=perturbation_size[sort_idx],
                cmap=color_map,
                norm=normalization,
                alpha=alpha,
                edgecolors=None,
            )

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def plot_perturbed_variables_1d(
    perturbations: xarray.Dataset, title: str = None, output_filename: PathLike = None,
):
    figure = pyplot.figure()
    figure.set_size_inches(11, 11 / 1.61803398875)
    if title is None:
        title = f'{len(perturbations["run"])} pertubation(s) of {len(perturbations["variable"])} variable(s)'
    figure.suptitle(title)

    variables = perturbations['variable'].values

    perturbations = perturbations['perturbations']
    for index, variable in enumerate(variables):
        axis = figure.add_subplot(len(variables), 1, index + 1)
        axis.title.set_text(f'{variable}')
        perturbed_var = perturbations.sel(variable=variable)
        axis.scatter(perturbed_var, perturbed_var * 0)
        min_val = perturbed_var.values.min().round(3)
        max_val = perturbed_var.values.max().round(3)
        axis.text(min_val, 0.02, f'min value = {min_val}', ha='left')
        axis.text(max_val, 0.02, f'max value = {max_val}', ha='right')

    if output_filename is not None:
        figure.savefig(output_filename, dpi=200, bbox_inches='tight')


def plot_perturbations(
    training_perturbations: xarray.Dataset,
    validation_perturbations: xarray.Dataset,
    runs: List[str],
    perturbation_types: List[str],
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
    plot_perturbed_variables_1d(
        training_perturbations,
        title=f'{len(training_perturbations["run"])} training pertubation(s) of {len(training_perturbations["variable"])} variable(s)',
        output_filename=output_directory / 'training_perturbations_1d.png'
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
    plot_perturbed_variables_1d(
        validation_perturbations,
        title=f'{len(validation_perturbations["run"])} validation pertubation(s) of {len(validation_perturbations["variable"])} variable(s)',
        output_filename=output_directory / 'validation_perturbations_1d.png'
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

            num_perturbations = len(runs)
            if 'original' in track_filenames.keys():
                runs = numpy.append(runs, 'original')
                perturbation_types = numpy.append(perturbation_types, 'original')

            figure = pyplot.figure()
            figure.set_size_inches(12, 12 / 1.61803398875)
            figure.suptitle(f'{num_perturbations} perturbations of storm track')

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
                storm = VortexTrack.from_file(track_filenames[run]).data
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


def plot_track_perturbations(
    perturbations: dict, storm_name: str = None, output_directory: PathLike = None,
):

    num_perturbations = len(perturbations)
    perturbation_keys = list(perturbations.keys())
    perturbation_types = ['ensemble'] * num_perturbations
    if 'original' in perturbation_keys:
        num_perturbations = num_perturbations - 1
        perturbation_keys.remove('original')
        perturbation_keys.append('original')
        perturbation_types[-1] = 'original'

    figure = pyplot.figure()
    figure.set_size_inches(12, 12)
    figure.suptitle(f'{num_perturbations} perturbations of {storm_name}')

    map_axis = figure.add_subplot(2, 1, 1)
    Vmax_axis = figure.add_subplot(2, 2, 3)
    Rmax_axis = figure.add_subplot(2, 2, 4)
    countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    unique_perturbation_types = numpy.unique(perturbation_types)
    encoded_perturbation_types = encode_categorical_values(
        perturbation_types, unique_values=unique_perturbation_types
    )
    linear_normalization = Normalize()
    colors = get_cmap('jet')(linear_normalization(encoded_perturbation_types))

    bounds = numpy.array([None, None, None, None])
    for index, run in enumerate(perturbation_keys):
        pkey = list(perturbations[run].keys())
        storm = VortexTrack.from_file(perturbations[run][pkey[0]]['fort22_filename']).data

        # Track
        points = storm.loc[:, ['longitude', 'latitude']].values.reshape(-1, 1, 2)
        segments = numpy.concatenate([points[:-1], points[1:]], axis=1)
        line_collection = LineCollection(
            segments,
            linewidths=numpy.concatenate(
                [[0], storm['radius_of_maximum_winds'] / 200,]  # maximum Rmax
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

        # Vmax
        Vmax_axis.plot(
            storm['datetime'],
            storm['max_sustained_wind_speed'],
            color=colors[index],
            linewidth=0.75 if run == 'original' else 0.25,
        )

        # Rmax
        Rmax_axis.plot(
            storm['datetime'],
            storm['radius_of_maximum_winds'],
            color=colors[index],
            linewidth=0.75 if run == 'original' else 0.25,
        )

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

    Rmax_axis.grid()
    Vmax_axis.grid()

    Rmax_axis.set_ylabel('$R_{max}$ [n mi]')
    Vmax_axis.set_ylabel('$V_{max}$ [kt]')

    Rmax_axis.set_xticklabels(Rmax_axis.get_xticklabels(), rotation=45)
    Vmax_axis.set_xticklabels(Vmax_axis.get_xticklabels(), rotation=45)

    Rmax_axis.set_title('Size')
    Vmax_axis.set_title('Intensity')
    map_axis.set_title('Trajectory')

    if output_directory is not None:
        figure.savefig(
            output_directory / f'{storm_name}_perturbations.png', dpi=200, bbox_inches='tight',
        )
