from abc import ABC, abstractmethod
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import List

from geopandas import GeoDataFrame
from matplotlib import cm, colors, pyplot
from matplotlib.lines import Line2D
import numpy
import pandas
from pandas import DataFrame
from pyproj import CRS, Geod
import shapely

from ensembleperturbation.parsing.adcirc import (
    ElevationStationOutput,
    parse_adcirc_outputs,
    VelocityStationOutput,
)
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing.comparison')

OBSERVATION_COLOR_MAP = cm.get_cmap('Blues')
MODEL_COLOR_MAP = cm.get_cmap('Reds')
ERROR_COLOR_MAP = cm.get_cmap('prism')

LINESTYLES = {'coldstart': ':', 'hotstart': '-'}

ADCIRC_VARIABLES = DataFrame(
    {
        'stations': ['fort.61.nc', 'fort.62.nc', 'fort.62.nc'],
        'model': ['fort.63.nc', 'fort.64.nc', 'fort.64.nc'],
        'max': ['maxele.63.nc', 'maxvel.63.nc', 'maxvel.63.nc'],
        'unit': ['m', 'm/s', 'm/s'],
        'name': ['zeta', 'u-vel', 'v-vel'],
    },
    index=['zeta', 'u', 'v'],
)


class ModelReferenceComparison(ABC):
    """
    abstraction of a comparison between reference data and modeled data
    """

    def __init__(
        self, input_directory: str, output_directory: str, variables: List[str],
    ):
        """
        :param input_directory: directory containing model inputs
        :param output_directory: directory containing model outputs
        :param variables: model variables to compare
        """

        if not isinstance(input_directory, Path):
            input_directory = Path(input_directory)
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)

        self.input_directory = input_directory
        self.output_directory = output_directory
        self.variables = variables

    @property
    @abstractmethod
    def values(self) -> GeoDataFrame:
        raise NotImplementedError

    @abstractmethod
    def plot_errors(self, show: bool = False):
        raise NotImplementedError

    @property
    @abstractmethod
    def rmses(self) -> dict:
        raise NotImplementedError


class ADCIRCReferenceComparison(ModelReferenceComparison):
    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        variables: List[str],
        stages: List[str] = None,
    ):
        """
        :param input_directory: directory containing model inputs
        :param output_directory: directory containing model outputs
        :param variables: model variables to compare
        :param stages: ADCIRC run stage (``coldstart``, ``hotstart``)
        """

        super(ADCIRCReferenceComparison, self).__init__(
            input_directory=input_directory,
            output_directory=output_directory,
            variables=variables,
        )

        self.stages = stages if stages is not None else ['coldstart', 'hotstart']

        self.fort14_filename = input_directory / 'fort.14'
        self.fort15_filename = input_directory / 'fort.15'

        self.runs = parse_adcirc_outputs(output_directory)

        self.crs = CRS.from_epsg(4326)
        ellipsoid = self.crs.datum.to_json_dict()['ellipsoid']
        self.geodetic = Geod(
            a=ellipsoid['semi_major_axis'], rf=ellipsoid['inverse_flattening']
        )

        first_run_stage = self.runs[list(self.runs)[0]][self.stages[0]]

        stations_basename = ADCIRC_VARIABLES.loc[self.variables[0]]['stations']
        if stations_basename in first_run_stage:
            self.stations = first_run_stage[stations_basename]
        else:
            self.stations = GeoDataFrame({'name': []}, geometry=[])

        self.mesh = first_run_stage[ADCIRC_VARIABLES.loc[self.variables[0]]['max']]

        self.stations = self.stations[
            self.stations.within(self.mesh.unary_union.convex_hull.buffer(0.01))
        ]
        self.stations.reset_index(drop=True, inplace=True)

    @property
    @lru_cache(maxsize=1)
    def station_mesh_vertices(self):
        nearest_mesh_vertices = []
        for station_index, station in self.stations.iterrows():
            nearest_mesh_vertex = shapely.ops.nearest_points(
                station.geometry, self.mesh.unary_union
            )[1]

            distance = self.geodetic.line_length(
                [station.geometry.x, nearest_mesh_vertex.x],
                [station.geometry.y, nearest_mesh_vertex.y],
            )
            mesh_index = self.mesh.cx[
                nearest_mesh_vertex.x, nearest_mesh_vertex.y
            ].index.item()
            nearest_mesh_vertices.append(
                GeoDataFrame(
                    {
                        'station': station['name'],
                        'station_x': station.geometry.x,
                        'station_y': station.geometry.y,
                        'distance': distance,
                    },
                    geometry=[nearest_mesh_vertex],
                    index=[mesh_index],
                )
            )
        if len(nearest_mesh_vertices) == 0:
            nearest_mesh_vertices.append(
                GeoDataFrame(
                    {'station': [], 'station_x': [], 'station_y': [], 'distance': [],},
                    geometry=[],
                    index=[],
                )
            )
        nearest_mesh_vertices = pandas.concat(nearest_mesh_vertices)
        nearest_mesh_vertices.reset_index(drop=True, inplace=True)
        return nearest_mesh_vertices

    @property
    @lru_cache(maxsize=1)
    def values(self) -> GeoDataFrame:
        observed_values = []
        for stage in self.stages:
            stations_filename = (
                self.output_directory
                / list(self.runs)[0]
                / stage
                / ADCIRC_VARIABLES.loc[self.variables[0]]['stations']
            )
            if stations_filename.exists():

                observed_values.append(
                    self.parse_stations(stations_filename, self.stations['name'])
                )
            else:
                LOGGER.warning(f'stations file not found at "{stations_filename}"')

        if len(observed_values) == 0:
            raise NoDataError('no station reference data provided')

        observed_values = pandas.concat(observed_values)

        values = []
        model_output_basename = ADCIRC_VARIABLES.loc[self.variables[0]]['model']
        for nearest_mesh_index, nearest_mesh_vertex in self.station_mesh_vertices.iterrows():
            station_name = nearest_mesh_vertex['station']

            station_values = None
            for run_name, stages in self.runs.items():
                run_values = []
                for stage, datasets in stages.items():
                    model_times = datasets[model_output_basename]['time']
                    modeled_values = {
                        variable_name: datasets[model_output_basename]['data'][
                            ADCIRC_VARIABLES.loc[variable_name]['name']
                        ]
                        for variable_name in self.variables
                    }

                    run_values.append(
                        GeoDataFrame(
                            {
                                'stage': stage,
                                'time': model_times,
                                'distance': nearest_mesh_vertex['distance'],
                                'geometry': [
                                    nearest_mesh_vertex.geometry for _ in model_times
                                ],
                                **{
                                    variable_name: variable_data[:, nearest_mesh_index]
                                    for variable_name, variable_data in modeled_values.items()
                                },
                            }
                        )
                    )

                    del model_times, modeled_values
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
                        station_values,
                        station_run_values,
                        how='left',
                        on=['stage', 'time', 'distance', 'geometry'],
                    )

            station_observed_values = observed_values[
                observed_values['station'] == station_name
            ]
            station_observed_values = station_observed_values[['time', *self.variables]]
            station_observed_values.columns = [
                'time',
                *(f'observed_{variable}' for variable in self.variables),
            ]

            station_values = pandas.merge(
                station_values, station_observed_values, how='left', on='time'
            )

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
        observed_values = values.iloc[:, [0, 2, *range(5, 5 + len(self.variables))]]

        errors = []
        for _, station in self.stations.iterrows():
            station_modeled_values = values[values['station'] == station['name']]
            station_distance = pandas.unique(station_modeled_values['distance'])[0]

            station_observed_values = observed_values[
                observed_values['station'] == station['name']
            ].iloc[:, 1:]
            station_observed_values.columns = ['time', *self.variables]

            station_errors = None
            for run_name in self.runs:
                run_modeled_values = station_modeled_values[
                    ['time', *(column for column in values.columns if run_name in column)]
                ]
                run_modeled_values.columns = ['time', *self.variables]

                run_errors = run_modeled_values - station_observed_values
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
                    station_errors = pandas.merge(
                        station_errors, run_errors, how='left', on=['time']
                    )

            station_errors.insert(0, 'station', station['name'])
            errors.append(station_errors)

        errors = pandas.concat(errors)
        errors.sort_values('time', inplace=True)
        errors.reset_index(drop=True, inplace=True)
        return errors

    @property
    @lru_cache(maxsize=1)
    def rmses(self) -> dict:
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

    @abstractmethod
    def parse_stations(self, filename: PathLike, station_names: List[str]) -> GeoDataFrame:
        raise NotImplementedError

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
                observed_values = stage_values[
                    [
                        'time',
                        *(column for column in stage_values.columns if 'observed_' in column),
                    ]
                ]
                observed_values.columns = ['time', *self.variables]

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
                            observed_values['time'],
                            observed_values[variable],
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
                                if column == f'{run_name}_' f'{variable}'
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


class ZetaComparison(ADCIRCReferenceComparison):
    """
    comparison between reference and ADCIRC-modeled sea-surface elevation
    """

    def __init__(self, input_directory: str, output_directory: str):
        """
        :param input_directory: directory containing model inputs
        :param output_directory: directory containing model outputs
        """

        super().__init__(
            input_directory, output_directory, ['zeta'], ['coldstart', 'hotstart']
        )

    def parse_stations(self, filename: PathLike, station_names: List[str]) -> GeoDataFrame:
        return ElevationStationOutput.read(filename, station_names)


class VelocityComparison(ADCIRCReferenceComparison):
    """
    comparison between reference and ADCIRC-modeled sea-surface horizontal velocity
    """

    def __init__(self, input_directory: str, output_directory: str):
        """
        :param input_directory: directory containing model inputs
        :param output_directory: directory containing model outputs
        """

        super().__init__(
            input_directory, output_directory, ['u', 'v'], ['coldstart', 'hotstart']
        )

    def parse_stations(self, filename: PathLike, station_names: List[str]) -> GeoDataFrame:
        return VelocityStationOutput.read(filename, station_names)


class NoDataError(Exception):
    pass


def insert_magnitude_components(
    dataframe: DataFrame,
    u: str = 'u',
    v: str = 'v',
    magnitude: str = 'magnitude',
    direction: str = 'direction',
    velocity_index: int = None,
    direction_index: int = None,
):
    if velocity_index is None:
        velocity_index = len(dataframe.columns)
    if direction_index is None:
        direction_index = velocity_index + 1
    dataframe.insert(velocity_index, magnitude, numpy.hypot(dataframe[u], dataframe[v]))
    dataframe.insert(direction_index, direction, numpy.arctan2(dataframe[u], dataframe[v]))
