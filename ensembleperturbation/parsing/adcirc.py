import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from os import PathLike
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Any, Collection, Mapping, Union

import geopandas
from geopandas import GeoDataFrame
from netCDF4 import Dataset
import numpy
import pandas
from pandas import DataFrame, Series
from shapely.geometry import Point

from ensembleperturbation.parsing.utilities import decode_time
from ensembleperturbation.perturbation.atcf import parse_vortex_perturbations
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing.adcirc')

ADCIRC_OUTPUT_DATA_VARIABLES = {
    # Elevation Time Series at Specified Elevation Recording Stations (fort.61)
    'fort.61.nc': ['station_name', 'zeta'],
    # Depth-averaged Velocity Time Series at Specified Velocity Recording
    # Stations (fort.62)
    'fort.62.nc': ['station_name', 'u-vel', 'v-vel'],
    # Elevation Time Series at All Nodes in the Model Grid (fort.63)
    'fort.63.nc': ['zeta'],
    # Depth-averaged Velocity Time Series at All Nodes in the Model Grid (
    # fort.64)
    'fort.64.nc': ['u-vel', 'v-vel'],
    # Hot Start Output (fort.67, fort.68)
    'fort.67.nc': ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel'],
    'fort.68.nc': ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel'],
    # Sea-level Pressure Time Series at All Nodes in the Model Grid (fort.73)
    'fort.73.nc': ['pressure'],
    # Surface Wind Velocity Time Series at All Nodes in the Model Grid (
    # fort.74)
    'fort.74.nc': ['windx', 'windy'],
    # Maximum Elevation at All Nodes in the Model Grid (maxele.63)
    'maxele.63.nc': ['zeta_max', 'time_of_zeta_max'],
    # Maximum Speed at All Nodes in the Model Grid (maxvel.63)
    'maxvel.63.nc': ['vel_max', 'time_of_vel_max'],
    # Minimum Sea-level Pressure at All Nodes in the Model Grid (minpr.63)
    'minpr.63.nc': ['pressure_min', 'time_of_pressure_min'],
    # Maximum Surface Wind Speed at All Nodes in the Model Grid (maxwvel.63)
    'maxwvel.63.nc': ['wind_max', 'time_of_wind_max'],
    # Maximum Radiation Surface Stress at All Nodes in the Model Grid (maxrs.63)
    'maxrs.63.nc': ['radstress_max', 'time_of_radstress_max'],
}

NODATA = -99999.0


def fort61_stations_zeta(filename: PathLike, station_names: [str] = None) -> GeoDataFrame:
    dataset = Dataset(filename)

    coordinate_variables = ['x', 'y']
    coordinates = numpy.stack([dataset[variable] for variable in coordinate_variables], axis=1)
    times = decode_time(dataset['time'])

    all_station_names = [
        station_name.tobytes().decode().strip().strip("'")
        for station_name in dataset['station_name']
    ]

    stations = []
    for station_name in station_names:
        station_index = all_station_names.index(station_name)
        station_coordinates = coordinates[station_index]
        station_point = Point(station_coordinates[0], station_coordinates[1])

        stations.append(
            GeoDataFrame(
                {
                    'time': times,
                    'zeta': dataset['zeta'][:, station_index],
                    'station': station_name,
                },
                geometry=[station_point for _ in times],
            )
        )

    return pandas.concat(stations)


def fort62_stations_uv(filename: PathLike, station_names: [str] = None) -> GeoDataFrame:
    dataset = Dataset(filename)

    coordinate_variables = ['x', 'y']
    coordinates = numpy.stack([dataset[variable] for variable in coordinate_variables], axis=1)
    times = decode_time(dataset['time'])

    all_station_names = [
        station_name.tobytes().decode().strip().strip("'")
        for station_name in dataset['station_name']
    ]

    stations = []
    for station_name in station_names:
        station_index = all_station_names.index(station_name)
        station_coordinates = coordinates[station_index]
        station_point = Point(station_coordinates[0], station_coordinates[1])

        stations.append(
            GeoDataFrame(
                {
                    'time': times,
                    'u': dataset['u-vel'][:, station_index],
                    'v': dataset['v-vel'][:, station_index],
                    'station': station_name,
                },
                geometry=[station_point for _ in times],
            )
        )

    return pandas.concat(stations)


def parse_adcirc_netcdf(filename: PathLike, variables: [str] = None) -> Union[dict, DataFrame]:
    """
    Parse ADCIRC output files

    :param filename: file path to ADCIRC NetCDF output
    :param variables: list of data variables to extract
    :return: parsed data
    """

    if not isinstance(filename, Path):
        filename = Path(filename)
    basename = filename.parts[-1]

    LOGGER.debug(f'opening "{filename.parts[-2:]}"')

    if variables is None:
        if basename in ADCIRC_OUTPUT_DATA_VARIABLES:
            variables = ADCIRC_OUTPUT_DATA_VARIABLES[basename]
        else:
            raise NotImplementedError(f'ADCIRC output file "{basename}" not implemented')

    dataset = Dataset(filename)

    data = {name: dataset[name] for name in variables}

    coordinate_variables = ['x', 'y']
    if 'depth' in dataset.variables:
        coordinate_variables += ['depth']
    coordinates = numpy.stack([dataset[variable] for variable in coordinate_variables], axis=1)

    times = decode_time(dataset['time'])

    if basename in ['fort.63.nc', 'fort.64.nc']:
        data = {'coordinates': coordinates, 'time': times, 'data': data}
    elif basename in ['fort.61.nc', 'fort.62.nc']:
        data = GeoDataFrame(
            {
                'name': [
                    station_name.tobytes().decode().strip().strip("'")
                    for station_name in dataset['station_name']
                ],
                'x': coordinates[:, 0],
                'y': coordinates[:, 1],
            },
            geometry=geopandas.points_from_xy(coordinates[:, 0], coordinates[:, 1]),
        )
    else:
        variables = {}
        for name, variable in data.items():
            if 'time_of' in name:
                variable = decode_time(variable, unit=dataset['time'].units)
            if variable.size > 0:
                variables[name] = numpy.squeeze(variable)
            else:
                LOGGER.warning(
                    f'array "{variable.name}" has invalid data shape "{variable.shape}"'
                )
                variables[name] = numpy.squeeze(
                    numpy.full(
                        [dimension if dimension > 0 else 1 for dimension in variable.shape],
                        fill_value=NODATA,
                    )
                )
        columns = dict(zip(coordinate_variables, coordinates.T))
        columns.update(variables)
        data = GeoDataFrame(
            columns, geometry=geopandas.points_from_xy(columns['x'], columns['y'])
        )
        data[data == NODATA] = numpy.nan

    LOGGER.debug(f'finished reading "{filename.parts[-2:]}"')

    return data


def async_parse_adcirc_netcdf(
    filename: PathLike, pickle_filename: PathLike, variables: [str] = None
) -> (Path, Path):
    if not isinstance(filename, Path):
        filename = Path(filename)
    if not isinstance(pickle_filename, Path):
        pickle_filename = Path(pickle_filename)

    data = parse_adcirc_netcdf(filename=filename, variables=variables)

    pickle_filename = pickle_data(data, pickle_filename)

    return filename, pickle_filename


def pickle_data(data: Any, filename: PathLike) -> Path:
    if not isinstance(filename, Path):
        filename = Path(filename)

    if isinstance(data, Mapping):
        filename.mkdir(parents=True, exist_ok=True)
        for variable, variable_data in data.items():
            pickle_data(variable_data, filename / variable)
    else:
        if isinstance(data, numpy.ndarray):
            filename = filename.parent / (filename.name + '.npy')
            if isinstance(data, numpy.ma.MaskedArray):
                data.dump(filename)
            else:
                numpy.save(filename, data)
        elif isinstance(data, DataFrame):
            filename = filename.parent / (filename.name + '.df')
            data.to_pickle(filename)
        else:
            filename = filename.parent / (filename.name + '.pickle')
            with open(filename, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)

    return filename


def parse_adcirc_outputs(
    directory: PathLike = None, file_data_variables: [str] = None
) -> {str: dict}:
    """
    Parse output from multiple ADCIRC runs.

    :param directory: directory containing run output directories
    :param file_data_variables: output files to parse
    :return: dictionary of file tree containing parsed data
    """

    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    if file_data_variables is None:
        file_data_variables = ADCIRC_OUTPUT_DATA_VARIABLES
    elif isinstance(file_data_variables, Collection):
        file_data_variables = {
            filename: ADCIRC_OUTPUT_DATA_VARIABLES[filename]
            for filename in file_data_variables
        }
    elif isinstance(file_data_variables, Mapping):
        file_data_variables = {
            filename: variables
            if variables is not None
            else ADCIRC_OUTPUT_DATA_VARIABLES[filename]
            for filename, variables in file_data_variables.items()
        }

    event_loop = asyncio.get_event_loop()
    process_pool = ProcessPoolExecutor()

    dataframes = {}
    with TemporaryDirectory() as temporary_directory:
        futures = []
        temporary_directory = Path(temporary_directory)
        for filename in directory.glob('**/*.nc'):
            if filename.name in file_data_variables:
                if filename.name in ['fort.63.nc', 'fort.64.nc']:
                    dataframes[filename] = parse_adcirc_netcdf(filename)
                else:
                    futures.append(
                        event_loop.run_in_executor(
                            process_pool,
                            partial(
                                async_parse_adcirc_netcdf,
                                filename=filename,
                                pickle_filename=temporary_directory / filename.name,
                            ),
                        )
                    )

        if len(futures) > 0:
            pickled_outputs = event_loop.run_until_complete(asyncio.gather(*futures))

            for input_filename, pickle_filename in pickled_outputs:
                if pickle_filename.is_dir():
                    dataframes[input_filename] = {}
                    for variable_pickle_filename in pickle_filename.iterdir():
                        with open(variable_pickle_filename) as pickle_file:
                            dataframes[input_filename][
                                variable_pickle_filename.stem
                            ] = pickle.load(pickle_file)
                else:
                    if pickle_filename.suffix == '.npy':
                        dataframes[input_filename] = numpy.load(pickle_filename)
                    elif pickle_filename.suffix == '.df':
                        dataframes[input_filename] = pandas.read_pickle(pickle_filename)
                    else:
                        with open(pickle_filename) as pickle_file:
                            dataframes[input_filename] = pickle.load(pickle_file)

    output_tree = {}
    for filename, dataframe in dataframes.items():
        parts = Path(str(filename).split(str(directory))[-1]).parts[1:]
        if parts[-1] not in file_data_variables:
            continue
        tree = output_tree
        for part_index in range(len(parts)):
            part = parts[part_index]
            if part_index < len(parts) - 1:
                if part not in tree:
                    tree[part] = {}
                tree = tree[part]
            else:
                tree[part] = dataframe

    return output_tree


def combine_outputs(
    directory: PathLike = None,
    bounds: (float, float, float, float) = None,
    maximum_depth: float = None,
    file_data_variables: {str: [str]} = None,
    output_filename: PathLike = None,
) -> DataFrame:
    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    if output_filename is not None:
        if not isinstance(output_filename, Path):
            output_filename = Path(output_filename)
        if output_filename.exists():
            os.remove(output_filename)

    runs_directory = directory / 'runs'
    if not runs_directory.exists():
        raise FileNotFoundError(f'runs directory does not exist at "{runs_directory}"')

    track_directory = directory / 'track_files'
    if not track_directory.exists():
        raise FileNotFoundError(f'track directory does not exist at "{track_directory}"')

    if file_data_variables is None:
        file_data_variables = ADCIRC_OUTPUT_DATA_VARIABLES
    elif isinstance(file_data_variables, Collection):
        file_data_variables = {
            filename: ADCIRC_OUTPUT_DATA_VARIABLES[filename]
            for filename in file_data_variables
        }
    elif isinstance(file_data_variables, Mapping):
        file_data_variables = {
            filename: variables
            if variables is not None
            else ADCIRC_OUTPUT_DATA_VARIABLES[filename]
            for filename, variables in file_data_variables.items()
        }

    # parse all the inputs using built-in parser
    vortex_perturbations = parse_vortex_perturbations(
        track_directory, output_filename=output_filename,
    )

    if output_filename is not None:
        key = 'vortex_perturbation_parameters'
        LOGGER.info(f'writing vortex perturbations to "{output_filename}/{key}"')
        vortex_perturbations.to_hdf(
            output_filename, key=key, mode='a', format='table', data_columns=True,
        )

    # parse all the outputs using built-in parser
    LOGGER.info(f'parsing from "{directory}"')
    output_data = parse_adcirc_outputs(
        directory=runs_directory, file_data_variables=file_data_variables,
    )

    if len(output_data) > 0:
        LOGGER.info(f'parsing results from {len(output_data)} runs')
    else:
        raise FileNotFoundError(f'could not find any output files in "{directory}"')

    # now assemble results into a single dataframe with:
    # rows -> index of a vertex in the mesh subset
    # columns -> name of perturbation, ( + x, y (lon, lat) and depth info)
    # values -> maximum elevation values ( + location and depths)
    subset = None
    dataframe = None
    variables = []
    for run_name, run_data in output_data.items():
        LOGGER.info(
            f'reading {len(run_data)} files from "{directory / run_name}": {list(run_data)}'
        )
        for result_filename, result_data in run_data.items():
            file_variables = file_data_variables[result_filename]

            variable_dataframe = result_data

            if isinstance(variable_dataframe, DataFrame):
                coordinate_variables = ['x', 'y']
                if 'depth' in variable_dataframe:
                    coordinate_variables.append('depth')

                if subset is None:
                    subset = Series(True, index=variable_dataframe.index)
                    if 'depth' in variable_dataframe:
                        if maximum_depth is not None:
                            subset &= variable_dataframe['depth'] < maximum_depth
                    if bounds is not None:
                        subset &= (variable_dataframe['x'] > bounds[0]) & (
                            variable_dataframe['x'] < bounds[2]
                        )
                        subset &= (variable_dataframe['y'] > bounds[1]) & (
                            variable_dataframe['y'] < bounds[3]
                        )
                    variable_dataframe = variable_dataframe.loc[subset]

                    LOGGER.info(
                        f'found {len(variable_dataframe)} records in "{result_filename}" ({variable_dataframe.columns})'
                    )

                try:
                    variable_dataframe = variable_dataframe[
                        coordinate_variables + file_variables
                    ]

                    if dataframe is None:
                        dataframe = variable_dataframe
                    else:
                        dataframe = dataframe.merge(
                            variable_dataframe, on=coordinate_variables, how='outer',
                        )
                except KeyError as error:
                    LOGGER.warning(error)
            else:
                LOGGER.warning(f'unable to parse data: {list(variable_dataframe)}')

            variables.extend(file_variables)

    LOGGER.info(f'parsed {len(variables)} variables')

    if output_filename is not None and dataframe is not None:
        for variable in variables:
            if variable in dataframe:
                LOGGER.info(f'writing "{output_filename}/{variable}"')
                variable_dataframe = dataframe[['x', 'y', 'depth', variable]]
                variable_dataframe.to_hdf(
                    output_filename, key=variable, mode='a', format='table', data_columns=True,
                )

    return dataframe
