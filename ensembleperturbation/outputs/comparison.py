from functools import lru_cache
from pathlib import Path
from typing import Callable

import geopandas
from geopandas import GeoDataFrame
from matplotlib import cm, colors, pyplot
from matplotlib.lines import Line2D
import numpy
import pandas
from pandas import DataFrame
from pyproj import CRS, Geod
import shapely

from ensembleperturbation.parsing.adcirc import (
    ADCIRC_VARIABLES,
    fort61_stations_zeta,
    fort62_stations_uv,
    parse_adcirc_outputs,
)

OBSERVATION_COLOR_MAP = cm.get_cmap('Blues')
MODEL_COLOR_MAP = cm.get_cmap('Reds')
ERROR_COLOR_MAP = cm.get_cmap('prism')

LINESTYLES = {'coldstart': ':', 'hotstart': '-'}

STATION_PARSERS = {
    'u': fort62_stations_uv,
    'v': fort62_stations_uv,
    'zeta': fort61_stations_zeta,
}


class StationComparison:
    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        variables: [str],
        stages: [str] = None,
        station_parsers: {str: Callable} = None,
        reference_label: str = None,
    ):
        if not isinstance(input_directory, Path):
            input_directory = Path(input_directory)
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.variables = variables
        self.station_parsers = (
            station_parsers
            if station_parsers is not None
            else {
                variable: station_parser
                for variable, station_parser in STATION_PARSERS.items()
                if variable in self.variables
            }
        )
        self.reference_label = reference_label if reference_label is not None else 'reference'

        self.fort14_filename = self.input_directory / 'fort.14'
        self.fort15_filename = self.input_directory / 'fort.15'

        # download_test_configuration(self.input_directory)

        self.runs = parse_adcirc_outputs(self.output_directory)

        self.stages = stages if stages is not None else ['coldstart', 'hotstart']

        self.crs = CRS.from_epsg(4326)
        ellipsoid = self.crs.datum.to_json_dict()['ellipsoid']
        self.geodetic = Geod(
            a=ellipsoid['semi_major_axis'], rf=ellipsoid['inverse_flattening']
        )

        first_run_stage = self.runs[list(self.runs)[0]][self.stages[0]]

        self.stations = first_run_stage[ADCIRC_VARIABLES.loc[self.variables[0]]['stations']]

        self.mesh = first_run_stage[ADCIRC_VARIABLES.loc[self.variables[0]]['max']]

        self.stations = self.stations[
            self.stations.within(self.mesh.unary_union.convex_hull.buffer(0.01))
        ]
        self.stations.reset_index(drop=True, inplace=True)

    @property
    @lru_cache(maxsize=1)
    def station_mesh_vertices(self):
        mesh_vertices = []
        for station_index, station in self.stations.iterrows():
            mesh_vertex = shapely.ops.nearest_points(station.geometry, self.mesh.unary_union)[
                1
            ]

            distance = self.geodetic.line_length(
                [station.geometry.x, mesh_vertex.x], [station.geometry.y, mesh_vertex.y]
            )
            mesh_index = self.mesh.cx[mesh_vertex.x, mesh_vertex.y].index.item()
            mesh_vertices.append(
                GeoDataFrame(
                    {
                        'station': station['name'],
                        'x': station.geometry.x,
                        'y': station.geometry.y,
                        'distance': distance,
                    },
                    geometry=[mesh_vertex],
                    index=[mesh_index],
                )
            )
        mesh_vertices = pandas.concat(mesh_vertices)
        mesh_vertices.reset_index(drop=True, inplace=True)
        return mesh_vertices

    @property
    @lru_cache(maxsize=1)
    def values(self) -> GeoDataFrame:
        reference_values = []
        for variable_info in self.variables:
            for stage in self.stages:
                station_parser = self.station_parsers[variable_info]

                stations_filename = (
                    self.output_directory
                    / list(self.runs)[0]
                    / stage
                    / ADCIRC_VARIABLES.loc[variable_info]['stations']
                )
                stage_reference_values = station_parser(
                    stations_filename, self.stations['name']
                )

                stage_reference_values.insert(1, 'stage', stage)
                reference_values.append(stage_reference_values)
        reference_values = pandas.concat(reference_values)

        values = []
        for nearest_mesh_index, nearest_mesh_vertex in self.station_mesh_vertices.iterrows():
            station_name = nearest_mesh_vertex['station']

            station_values = None
            for run_name, stages in self.runs.items():
                run_values = []
                for stage, datasets in stages.items():
                    stage_values = None
                    for variable_name in self.variables:
                        variable_info = ADCIRC_VARIABLES.loc[variable_name]
                        times = datasets[variable_info['model']]['time']
                        variable_data = GeoDataFrame(
                            {
                                'stage': stage,
                                'time': times,
                                'distance': nearest_mesh_vertex['distance'],
                                'geometry': [nearest_mesh_vertex.geometry for _ in times],
                                variable_name: datasets[variable_info['model']]['data'][
                                    variable_info['name']
                                ][:, nearest_mesh_index],
                            }
                        )

                        if stage_values is None:
                            stage_values = variable_data
                        else:
                            stage_values = pandas.merge(
                                stage_values, variable_data, how='left'
                            )
                    run_values.append(stage_values)

                run_values = pandas.concat(run_values)
                run_values.columns = [
                    f'{run_name}_{column}' if column in self.variables else column
                    for column in run_values.columns
                ]

                if station_values is None:
                    station_values = run_values
                else:
                    station_run_values = run_values[
                        [
                            'stage',
                            'time',
                            'distance',
                            'geometry',
                            *(f'{run_name}_{variable}' for variable in self.variables),
                        ]
                    ]
                    station_values = pandas.merge(
                        station_values, station_run_values, how='left'
                    )

            station_reference_values = reference_values[
                reference_values['station'] == station_name
            ]
            station_reference_values = station_reference_values[['time', *self.variables]]
            station_reference_values.columns = [
                'time',
                *(f'{self.reference_label}_{variable}' for variable in self.variables),
            ]

            station_values = pandas.merge(station_values, station_reference_values, how='left')

            station_values.insert(0, 'station', station_name)
            values.append(station_values)

        values = pandas.concat(values)
        values.sort_values('time', inplace=True)

        values = values.iloc[
            :,
            [
                *range(5),
                *range(-len(self.variables), 0),
                *range(5, len(values.columns) - len(self.variables)),
            ],
        ]
        values.reset_index(drop=True, inplace=True)
        return values

    @property
    @lru_cache(maxsize=1)
    def errors(self) -> GeoDataFrame:
        values = self.values
        reference_values = values.iloc[:, [0, 2, *range(5, 5 + len(self.variables))]]

        errors = []
        for _, station in self.stations.iterrows():
            station_modeled_values = values[values['station'] == station['name']]
            station_distance = pandas.unique(station_modeled_values['distance'])[0]

            station_reference_values = reference_values[
                reference_values['station'] == station['name']
            ].iloc[:, 1:]
            station_reference_values.columns = ['time', *self.variables]

            station_errors = None
            for run_name in self.runs:
                run_modeled_values = station_modeled_values[
                    ['time', *(column for column in values.columns if run_name in column)]
                ]
                run_modeled_values.columns = ['time', *self.variables]

                run_errors = run_modeled_values - station_reference_values
                del run_modeled_values

                run_errors.columns = ['time_difference', *self.variables]
                run_errors.columns = [
                    f'{run_name}_{column}' if column in self.variables else column
                    for column in run_errors.columns
                ]

                run_errors.insert(0, 'time', station_modeled_values['time'])
                run_errors.insert(1, 'stage', station_modeled_values['stage'])
                run_errors.insert(2, 'distance', station_distance)

                if station_errors is None:
                    station_errors = run_errors
                else:
                    run_errors = run_errors[
                        ['time', *(f'{run_name}_{variable}' for variable in self.variables)]
                    ]
                    station_errors = pandas.merge(station_errors, run_errors, how='left')

            station_errors.insert(0, 'station', station['name'])
            errors.append(station_errors)

        errors = pandas.concat(errors)
        errors.sort_values('time', inplace=True)
        errors.reset_index(drop=True, inplace=True)
        return errors

    @property
    @lru_cache(maxsize=1)
    def rmses(self) -> {}:
        errors = self.errors
        rmses = []
        for _, station in self.stations.iterrows():
            station_errors = errors[errors['station'] == station['name']]
            station_distance = pandas.unique(station_errors['distance'])[0]
            station_rmses = []
            for run_name in self.runs:
                run_errors = station_errors[
                    [
                        'stage',
                        'time',
                        *(column for column in errors.columns if run_name in column),
                    ]
                ]
                run_rmses = []
                for stage in self.stages:
                    stage_errors = run_errors[run_errors['stage'] == stage]
                    stage_rmses = {'stage': stage}
                    for column in stage_errors.iloc[:, 2:]:
                        variable = column.replace(f'{run_name}_', '')
                        stage_rmses[variable] = numpy.sqrt((stage_errors[column] ** 2).mean())
                    run_rmses.append(stage_rmses)
                run_rmses = DataFrame.from_dict(
                    dict(zip(range(len(run_rmses)), run_rmses)), orient='index'
                )
                run_rmses.insert(0, 'run', run_name)
                station_rmses.append(run_rmses)

            station_rmses = pandas.concat(station_rmses)
            station_rmses = GeoDataFrame(
                station_rmses, geometry=[station.geometry for _ in range(len(station_rmses))]
            )

            station_rmses.insert(1, 'station', station['name'])
            station_rmses.insert(1, 'distance', station_distance)
            rmses.append(station_rmses)

        rmses = pandas.concat(rmses)
        rmses.reset_index(drop=True, inplace=True)
        return rmses

    def plot_values(self, show: bool = False):
        values = self.values

        run_index_normalizer = colors.Normalize(0, len(self.runs))

        figure = pyplot.figure()
        figure.suptitle('stations')
        sharing_axis = None

        value_axes = {}
        for station_index, station in self.stations.iterrows():
            station_axes = {}
            for variable_index, variable in enumerate(self.variables):
                axis = figure.add_subplot(
                    len(self.stations) * len(self.variables),
                    1,
                    station_index * len(self.variables) + (variable_index + 1),
                    sharex=sharing_axis,
                )
                if sharing_axis is None:
                    sharing_axis = axis
                station_axes[variable] = axis

            value_axes[station['name']] = station_axes

        for station_index, station in self.stations.iterrows():
            axes = value_axes[station['name']]
            station_values = values[values['station'] == station['name']]
            for stage in self.stages:
                stage_values = station_values[station_values['stage'] == stage]
                reference_values = stage_values[
                    [
                        'time',
                        *(
                            column
                            for column in stage_values.columns
                            if f'{self.reference_label}_' in column
                        ),
                    ]
                ]
                reference_values.columns = ['time', *self.variables]

                for run_index, run_name in enumerate(self.runs):
                    observation_color = OBSERVATION_COLOR_MAP(run_index_normalizer(run_index))
                    model_color = MODEL_COLOR_MAP(run_index_normalizer(run_index))

                    modeled_values = stage_values[
                        [
                            'time',
                            *(
                                column
                                for column in stage_values.columns
                                if f'{run_name}_' in column
                            ),
                        ]
                    ]
                    modeled_values.columns = ['time', *self.variables]

                    for variable, axis in axes.items():
                        axis.plot(
                            reference_values['time'],
                            reference_values[variable],
                            color=observation_color,
                            linestyle=LINESTYLES[stage],
                        )
                        axis.plot(
                            modeled_values['time'],
                            modeled_values[variable],
                            color=model_color,
                            linestyle=LINESTYLES[stage],
                        )

        handles = [
            Line2D([0], [0], color='b', label='observation'),
            Line2D([0], [0], color='r', label='model'),
            Line2D([0], [0], color='k', linestyle=LINESTYLES['coldstart'], label='coldstart'),
            Line2D([0], [0], color='k', linestyle=LINESTYLES['hotstart'], label='hotstart'),
        ]

        for station_name, axes in value_axes.items():
            for variable, axis in axes.items():
                axis.set_title(f'station {station_name} {variable}', loc='left')
                axis.hlines([0], *axis.get_xlim(), color='k', linestyle='--')
                axis.set_ylabel(f'{variable} ({ADCIRC_VARIABLES.loc[variable]["unit"]})')
                axis.legend(handles=handles)

        if show:
            pyplot.show()

    def plot_errors(self, show: bool = False):
        errors = self.errors

        station_index_normalizer = colors.Normalize(0, len(self.stations))

        figure = pyplot.figure()
        figure.suptitle('errors')
        sharing_axis = None

        error_axes = {}
        for variable_index, variable in enumerate(self.variables):
            axis = figure.add_subplot(
                len(self.variables), 1, variable_index + 1, sharex=sharing_axis
            )
            if sharing_axis is None:
                sharing_axis = axis

            error_axes[variable] = axis

        for variable, axis in error_axes.items():
            for station_index, station in self.stations.iterrows():
                error_color = ERROR_COLOR_MAP(station_index_normalizer(station_index))

                station_errors = errors[errors['station'] == station['name']]

                for run_index, run_name in enumerate(self.runs):
                    variable_errors = station_errors[
                        [
                            'time',
                            'stage',
                            *(
                                column
                                for column in errors
                                if column == f'{run_name}_{variable}'
                            ),
                        ]
                    ]
                    variable_errors.columns = ['time', 'stage', variable]
                    for stage in self.stages:
                        stage_errors = variable_errors[variable_errors['stage'] == stage]
                        axis.plot(
                            stage_errors['time'],
                            stage_errors[variable],
                            color=error_color,
                            linestyle=LINESTYLES[stage],
                        )

        handles = [
            *(
                Line2D(
                    [0],
                    [0],
                    color=ERROR_COLOR_MAP(station_index_normalizer(station_index)),
                    label=f'station {station["name"]}',
                )
                for station_index, station in self.stations.iterrows()
            ),
            Line2D([0], [0], color='k', linestyle=LINESTYLES['coldstart'], label='coldstart'),
            Line2D([0], [0], color='k', linestyle=LINESTYLES['hotstart'], label='hotstart'),
        ]

        for variable, axis in error_axes.items():
            axis.set_title(f'{variable} error', loc='left')
            axis.hlines([0], *axis.get_xlim(), color='k', linestyle='--')
            axis.set_ylabel(f'{variable} error ' f'({ADCIRC_VARIABLES.loc[variable]["unit"]})')
            axis.legend(handles=handles)

        if show:
            pyplot.show()

    def plot_rmse(self, show: bool = False):
        rmses = self.rmses

        station_index_normalizer = colors.Normalize(0, len(self.stations))

        figure = pyplot.figure()
        figure.suptitle('RMSE')
        sharing_axis = None

        axes = {}
        for variable_index, variable in enumerate(self.variables):
            axis = figure.add_subplot(len(self.variables), 1, variable_index + 1)
            if sharing_axis is None:
                sharing_axis = axis
            axes[variable] = axis

        for variable in self.variables:
            variable_rmses = rmses[
                [
                    'run',
                    'station',
                    'stage',
                    *(column for column in rmses.columns if column == variable),
                ]
            ]

            axis = axes[variable]
            for stage in self.stages:
                stage_rmses = variable_rmses[variable_rmses['stage'] == stage]
                for station_index, station in self.stations.iterrows():
                    station_rmses = stage_rmses[stage_rmses['station'] == station['name']]
                    axis.plot(
                        station_rmses['run'],
                        station_rmses[variable],
                        color=ERROR_COLOR_MAP(station_index_normalizer(station_index)),
                        linestyle=LINESTYLES[stage],
                        label=f'{stage} {variable}',
                    )

        handles = [
            *(
                Line2D(
                    [0],
                    [0],
                    color=ERROR_COLOR_MAP(station_index_normalizer(station_index)),
                    label=f'station {station["name"]}',
                )
                for station_index, station in self.stations.iterrows()
            ),
            Line2D([0], [0], color='k', linestyle=LINESTYLES['coldstart'], label='coldstart'),
            Line2D([0], [0], color='k', linestyle=LINESTYLES['hotstart'], label='hotstart'),
        ]

        for variable, axis in axes.items():
            axis.set_xlabel('run')
            axis.set_ylabel(f'{variable} ({ADCIRC_VARIABLES.loc[variable]["unit"]})')
            axis.legend(handles=handles)

        if show:
            pyplot.show()


class ObservationStationComparison(StationComparison):
    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        variables: [str],
        stages: [str] = None,
    ):
        super().__init__(
            input_directory, output_directory, variables, stages, reference_label='observed'
        )


class VirtualStationComparison(StationComparison):
    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        variables: [str],
        stations_filename: str,
        stages: [str] = None,
        method: Callable = numpy.mean,
    ):
        station_parsers = {variable: None for variable in self.variables}
        super().__init__(
            input_directory,
            output_directory,
            variables,
            stages,
            station_parsers,
            reference_label='virtual',
        )
        stations = geopandas.read_file(stations_filename)
        if 'name' not in stations.columns:
            stations.insert(0, 'name', range(len(stations)))
        self.stations = stations


def vector_magnitude(vectors: numpy.array):
    if not isinstance(vectors, numpy.ndarray):
        vectors = numpy.array(vectors)
    if len(vectors.shape) < 2:
        vectors = numpy.expand_dims(vectors, axis=0)
    return numpy.stack(
        [
            numpy.hypot(vectors[:, 0], vectors[:, 1]),
            numpy.arctan2(vectors[:, 0], vectors[:, 1]),
        ],
        axis=1,
    )
