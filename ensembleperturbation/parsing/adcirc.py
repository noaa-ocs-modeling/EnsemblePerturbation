import concurrent.futures
import os
from os import PathLike
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Any, Collection, List, Mapping, Union

import geopandas
from geopandas import GeoDataFrame
import netCDF4
import numpy
import pandas
from pandas import DataFrame, Series
from shapely.geometry import Point
import xarray
from xarray import DataArray

from ensembleperturbation.parsing.utilities import decode_time
from ensembleperturbation.perturbation.atcf import parse_vortex_perturbations
from ensembleperturbation.utilities import get_logger, ProcessPoolExecutorStackTraced

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
    dataset = netCDF4.Dataset(filename)

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
    dataset = netCDF4.Dataset(filename)

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

    LOGGER.debug(f'opening "{"/".join(filename.parts[-2:])}"')

    if variables is None:
        if basename in ADCIRC_OUTPUT_DATA_VARIABLES:
            variables = ADCIRC_OUTPUT_DATA_VARIABLES[basename]
        else:
            raise NotImplementedError(f'ADCIRC output file "{basename}" not implemented')

    dataset = netCDF4.Dataset(filename)

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
                LOGGER.debug(
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

    LOGGER.debug(f'finished reading "{"/".join(filename.parts[-2:])}"')

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
    directory: PathLike = None, file_data_variables: [str] = None, parallel: bool = False,
) -> {str: dict}:
    """
    Parse output from multiple ADCIRC runs.

    :param directory: directory containing run output directories
    :param file_data_variables: output files to parse
    :param parallel: parse outputs concurrently
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

    dataframes = {}
    with TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        process_pool = ProcessPoolExecutorStackTraced()
        futures = []

        for filename in directory.glob('**/*.nc'):
            if filename.name in file_data_variables:
                if not parallel or filename.name in ['fort.63.nc', 'fort.64.nc']:
                    dataframes[filename] = parse_adcirc_netcdf(filename)
                else:
                    futures.append(
                        process_pool.submit(
                            async_parse_adcirc_netcdf,
                            filename=filename,
                            pickle_filename=temporary_directory / filename.name,
                        ),
                    )

        if len(futures) > 0:
            for completed_future in concurrent.futures.as_completed(futures):
                input_filename, pickle_filename = completed_future.result()

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
    parallel: bool = False,
) -> {str: DataFrame}:
    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    if maximum_depth is not None and not isinstance(maximum_depth, float):
        maximum_depth = float(maximum_depth)

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
    vortex_perturbations = parse_vortex_perturbations(track_directory)

    if output_filename is not None:
        key = 'vortex_perturbation_parameters'
        LOGGER.info(f'writing vortex perturbations to "{output_filename}/{key}"')
        vortex_perturbations.to_hdf(
            output_filename, key=key, mode='a', format='table', data_columns=True,
        )

    # parse all the outputs using built-in parser
    LOGGER.info(f'parsing from "{directory}"')
    output_data = parse_adcirc_outputs(
        directory=runs_directory, file_data_variables=file_data_variables, parallel=parallel,
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
    variables_data = {}
    for run_name, run_data in output_data.items():
        LOGGER.info(
            f'reading {len(run_data)} files from "{directory / run_name}": {list(run_data)}'
        )

        for result_filename, result_data in run_data.items():
            variables = file_data_variables[result_filename]

            if isinstance(result_data, DataFrame):
                num_records = len(result_data)

                coordinate_variables = ['x', 'y']
                if 'depth' in result_data:
                    coordinate_variables.append('depth')

                if subset is None:
                    subset = Series(True, index=result_data.index)
                    if 'depth' in result_data:
                        if maximum_depth is not None:
                            subset &= result_data['depth'] < maximum_depth
                    if bounds is not None:
                        subset &= (result_data['x'] > bounds[0]) & (
                            result_data['x'] < bounds[2]
                        )
                        subset &= (result_data['y'] > bounds[1]) & (
                            result_data['y'] < bounds[3]
                        )
                    result_data = result_data.loc[subset]
                    LOGGER.debug(
                        f'subsetting: {num_records} nodes -> {len(result_data)} nodes'
                    )

                LOGGER.debug(
                    f'found {len(variables)} variables over {len(result_data)} nodes in "{run_name}/{result_filename}"'
                )

                for variable_name in variables:
                    variable_data = result_data[coordinate_variables + [variable_name]].rename(
                        columns={variable_name: run_name}
                    )

                    hdf5_variable = variable_name.replace('-', '_')
                    if hdf5_variable in variables_data:
                        variables_data[hdf5_variable] = variables_data[hdf5_variable].merge(
                            variable_data, on=coordinate_variables, how='outer', copy=False,
                        )
                    else:
                        variables_data[hdf5_variable] = variable_data
            else:
                coordinates = DataArray(
                    result_data['coordinates'][:],
                    dims=['node', 'coordinate'],
                    coords={'node': numpy.arange(len(result_data['coordinates']))},
                )

                for variable_name, variable_data in result_data['data'].items():
                    if variable_name not in variables_data:
                        variables_data[variable_name] = []
                    data_array = DataArray(
                        variable_data[:],
                        dims=variable_data.dimensions,
                        coords={
                            'time': result_data['time'],
                            'node': coordinates.coords['node'],
                            'x': coordinates[:, 0],
                            'y': coordinates[:, 1],
                            'z': coordinates[:, 2],
                        },
                        name=run_name,
                    )

                    if maximum_depth is not None:
                        data_array = data_array.isel(node=coordinates[:, 2] < -maximum_depth)

                    variables_data[variable_name].append(data_array)

    if output_filename is not None and len(variables_data) > 0:
        LOGGER.info(f'parsed {len(variables_data)} variables: {list(variables_data)}')

        data_arrays = {}
        for variable_name, variable_data in variables_data.items():
            if isinstance(variable_data, DataFrame):
                duplicate_indices = variable_data.index[variable_data.index.duplicated()]
                if len(duplicate_indices) > 0:
                    LOGGER.warning(
                        f'{len(duplicate_indices)} duplicate indices found: {duplicate_indices}'
                    )
                    variable_data.drop(duplicate_indices, axis=0, inplace=True)

                duplicate_columns = variable_data.columns[variable_data.columns.duplicated()]
                if len(duplicate_columns) > 0:
                    LOGGER.warning(
                        f'{len(duplicate_columns)} duplicate columns found: {duplicate_columns}'
                    )
                    variable_data.drop(duplicate_columns, axis=1, inplace=True)

                LOGGER.info(
                    f'writing {variable_name} over {len(variable_data)} nodes to "{output_filename}/{variable_name}"'
                )
                variable_data.to_hdf(
                    output_filename,
                    key=variable_name,
                    mode='a',
                    format='table',
                    data_columns=True,
                )
            elif isinstance(variable_data, List):
                # combine N-dimensional xarrays for this variable into an additional `run` dimension specifying the run name
                variable_data = xarray.concat(
                    variable_data,
                    dim=DataArray(
                        [data_array.name for data_array in variable_data],
                        dims=['run'],
                        name='run',
                    ),
                )
                data_arrays[variable_name] = variable_data

        if len(data_arrays) > 0:
            dataset = xarray.Dataset(data_vars=data_arrays)
            dataset.to_netcdf(output_filename.parent / (output_filename.stem + '.nc'))

    return variables_data
