import asyncio
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from pathlib import Path
from typing import Union

import geopandas
from geopandas import GeoDataFrame
from netCDF4 import Dataset
import numpy
import pandas
from pandas import DataFrame, Series
from shapely.geometry import Point

from ensembleperturbation.parsing.utilities import decode_time
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

    return data


def parse_adcirc_output(
    directory: PathLike, file_data_variables: [str] = None
) -> {str: Union[dict, DataFrame]}:
    """
    Parse ADCIRC output files

    :param directory: path to directory containing ADCIRC output files in
    NetCDF format
    :param file_data_variables: output files to parsing
    :return: dictionary of output data
    """

    if not isinstance(directory, Path):
        directory = Path(directory)

    if file_data_variables is None:
        file_data_variables = ADCIRC_OUTPUT_DATA_VARIABLES
    else:
        file_data_variables = {
            filename: ADCIRC_OUTPUT_DATA_VARIABLES[filename]
            for filename in file_data_variables
        }

    output_data = {}
    for output_filename in directory.glob('*.nc'):
        basename = output_filename.parts[-1]
        if basename in file_data_variables:
            output_data[basename] = parse_adcirc_netcdf(
                output_filename, file_data_variables[basename]
            )

    return output_data


def async_parse_adcirc_netcdf(filename: Path, part: str, variables: [str] = None):
    LOGGER.info(f'starting reading "{filename.parts[-2:]}"')
    output = parse_adcirc_netcdf(filename=filename, variables=variables)
    LOGGER.info(f'finished reading "{filename.parts[-2:]}"')
    return output, part


def parse_adcirc_outputs(
    directory: PathLike = None, file_data_variables: [str] = None
) -> {str: dict}:
    """
    Parse output from multiple ADCIRC runs.

    :param directory: directory containing run output directories
    :param file_data_variables: output files to parsing
    :return: dictionary of file tree containing parsed data
    """

    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    if file_data_variables is None:
        file_data_variables = ADCIRC_OUTPUT_DATA_VARIABLES
    else:
        file_data_variables = {
            filename: ADCIRC_OUTPUT_DATA_VARIABLES[filename]
            for filename in file_data_variables
        }

    event_loop = asyncio.get_event_loop()
    process_pool = ProcessPoolExecutor()

    output_datasets = {}
    for filename in directory.glob('**/*.nc'):
        parts = Path(str(filename).split(str(directory))[-1]).parts[1:]
        if parts[-1] not in file_data_variables:
            continue
        LOGGER.debug(f'read file "{filename}"')
        tree = output_datasets
        for part_index in range(len(parts)):
            part = parts[part_index]
            if part_index < len(parts) - 1:
                if part not in tree:
                    tree[part] = {}
                tree = tree[part]
            else:
                try:
                    event_loop.create_task(
                        event_loop.run_in_executor(
                            process_pool, async_parse_adcirc_netcdf, filename, part
                        )
                    )
                except Exception as error:
                    LOGGER.warning(f'{error.__class__.__name__} - {error}')

            for task in asyncio.as_completed(asyncio.all_tasks(event_loop)):
                result, part = event_loop.run_until_complete(task)
                tree[part] = result

    return output_datasets


def combine_outputs(
    directory: PathLike = None,
    bounds: (float, float, float, float) = None,
    maximum_depth: float = None,
    output_filename: PathLike = None,
) -> DataFrame:
    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    # define the output file type and variable name interested name
    output_filetypes = {
        'maxele.63.nc': 'zeta_max',
    }

    # parse all the outputs using built-in parser
    LOGGER.info(f'parsing {output_filetypes} from "{directory}"')
    output_data = parse_adcirc_outputs(
        directory=directory, file_data_variables=output_filetypes.keys(),
    )

    if len(output_data) == 0:
        raise FileNotFoundError(f'could not find any output files in "{directory}"')
    else:
        LOGGER.info(f'building dataframe from {len(output_data)} perturbations')

    # now assemble results into a single dataframe with:
    # rows -> index of a vertex in the mesh subset
    # columns -> name of perturbation, ( + x, y (lon, lat) and depth info)
    # values -> maximum elevation values ( + location and depths)
    subset = None
    dataframe = None
    for pertubation_index, perturbation in enumerate(output_data):
        for variable in output_data[perturbation]:
            variable_dataframe = output_data[perturbation][variable]
            if subset is None:
                subset = Series(True, index=variable_dataframe.index)
                if maximum_depth is not None:
                    subset &= variable_dataframe['depth'] < maximum_depth
                if bounds is not None:
                    subset &= (variable_dataframe['x'] > bounds[0]) & (
                        variable_dataframe['x'] < bounds[2]
                    )
                    subset &= (variable_dataframe['y'] > bounds[1]) & (
                        variable_dataframe['y'] < bounds[3]
                    )
                dataframe = variable_dataframe[['x', 'y', 'depth']][subset]
            dataframe.insert(
                2, perturbation, variable_dataframe[output_filetypes[variable]][subset], True
            )

            if output_filename is not None:
                LOGGER.info(f'writing to "{output_filename}"')
                dataframe.to_hdf(
                    output_filename,
                    key=output_filetypes[variable],
                    mode='a',
                    format='table',
                    data_columns=True,
                )

    return dataframe
