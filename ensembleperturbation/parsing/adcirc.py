from abc import ABC, abstractmethod
import os
from os import PathLike
from pathlib import Path
import pickle
from typing import Any, Collection, Mapping, Union

import dask
import geopandas
from geopandas import GeoDataFrame
import numpy
import pandas
from pandas import DataFrame
from shapely.geometry import Point
import xarray
from xarray import DataArray, Dataset

from ensembleperturbation.perturbation.atcf import parse_vortex_perturbations
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing.adcirc')


class AdcircOutput(ABC):
    filename: PathLike
    variables: [str]
    drop_variables: [str] = ['neta']
    nodata: float = -99999.0

    @classmethod
    @abstractmethod
    def read(cls, filename: PathLike, names: [str] = None) -> Union[DataFrame, DataArray]:
        raise NotImplementedError

    @classmethod
    def async_read(
        cls, filename: PathLike, pickle_filename: PathLike, variables: [str] = None
    ) -> (Path, Path):
        if not isinstance(filename, Path):
            filename = Path(filename)
        if not isinstance(pickle_filename, Path):
            pickle_filename = Path(pickle_filename)

        data = cls.read(filename=filename, names=variables)
        pickle_filename = pickle_data(data, pickle_filename)

        return filename, pickle_filename

    @classmethod
    def read_directory(
        cls, directory: PathLike, variables: [str] = None, parallel: bool = False
    ) -> Dataset:
        """
        Compile a dataset from output files in the given directory.

        :param directory: directory containing output files
        :param variables: variables to return
        :param parallel: load data concurrently with Dask
        :return: dataset of output data
        """

        if not isinstance(directory, Path):
            directory = Path(directory)

        if variables is None:
            variables = cls.variables

        filenames = list(directory.glob(f'**/{cls.filename}'))

        drop_variables = cls.drop_variables

        with xarray.open_dataset(
            filenames[0], drop_variables=drop_variables
        ) as sample_dataset:
            drop_variables.extend(
                variable_name
                for variable_name in sample_dataset.variables
                if variable_name not in variables
                and variable_name not in ['node', 'time', 'x', 'y', 'depth']
            )

        dataset = xarray.open_mfdataset(
            filenames,
            drop_variables=drop_variables,
            combine='nested',
            concat_dim=xarray.DataArray(
                [filename.parent.name for filename in filenames], dims=['run'], name='run',
            ),
            parallel=parallel,
        )

        if 'depth' in dataset:
            dataset = dataset.assign_coords(
                {'depth': dataset['depth'].isel(run=0).drop_vars('run')}
            )

        if 'node' not in dataset:
            dataset = dataset.assign_coords({'node': dataset['node']})

        return dataset[variables]

    @classmethod
    @abstractmethod
    def subset(
        cls, dataset: Dataset, bounds: (float, float, float, float) = None, **kwargs,
    ) -> Dataset:
        raise NotImplementedError


class TimeSeriesOutput(ABC):
    pass


class StationTimeSeriesOutput(AdcircOutput, TimeSeriesOutput, ABC):
    @classmethod
    @abstractmethod
    def read(cls, filename: PathLike, names: [str] = None) -> GeoDataFrame:
        raise NotImplementedError

    @classmethod
    def subset(
        cls, dataset: Dataset, bounds: (float, float, float, float) = None, **kwargs,
    ) -> Dataset:
        subset = ~dataset['station'].isnull()

        if bounds is not None:
            LOGGER.debug(f'filtering within bounds {bounds}')
            if bounds[0] is not None:
                subset = xarray.ufuncs.logical_and(subset, dataset['x'] > bounds[0])
            if bounds[2] is not None:
                subset = xarray.ufuncs.logical_and(subset, dataset['x'] < bounds[2])
            if bounds[1] is not None:
                subset = xarray.ufuncs.logical_and(subset, dataset['y'] > bounds[1])
            if bounds[3] is not None:
                subset = xarray.ufuncs.logical_and(subset, dataset['y'] < bounds[3])

        return subset


class ElevationStationOutput(StationTimeSeriesOutput):
    """ Elevation Time Series at Specified Elevation Recording Stations (fort.61) """

    filename = 'fort.61.nc'
    variables = ['station_name', 'zeta']

    @classmethod
    def read(cls, filename: PathLike, names: [str] = None) -> GeoDataFrame:
        dataset = xarray.open_dataset(filename, drop_variables=cls.drop_variables)

        coordinate_variables = ['x', 'y']
        coordinates = numpy.stack(
            [dataset[variable] for variable in coordinate_variables], axis=1
        )
        times = dataset['time']

        all_station_names = [
            station_name.tobytes().decode().strip().strip("'")
            for station_name in dataset['station_name']
        ]

        stations = []
        for station_name in names:
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


class VelocityStationOutput(StationTimeSeriesOutput):
    """ Depth-averaged Velocity Time Series at Specified Velocity Recording Stations (fort.62) """

    filename = 'fort.62.nc'
    variables = ['station_name', 'u-vel', 'v-vel']

    @classmethod
    def read(cls, filename: PathLike, names: [str] = None) -> GeoDataFrame:
        dataset = xarray.open_dataset(filename, drop_variables=cls.drop_variables)

        coordinate_variables = ['x', 'y']
        coordinates = numpy.stack(
            [dataset[variable] for variable in coordinate_variables], axis=1
        )
        times = dataset['time']

        all_station_names = [
            station_name.tobytes().decode().strip().strip("'")
            for station_name in dataset['station_name']
        ]

        stations = []
        for station_name in names:
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


class FieldOutput(AdcircOutput, ABC):
    @classmethod
    def read(cls, filename: PathLike, names: [str] = None) -> Union[DataFrame, DataArray]:
        """
        Parse ADCIRC output files

        :param filename: file path to ADCIRC NetCDF output
        :param names: list of data variables to extract
        :return: parsed data
        """

        if not isinstance(filename, Path):
            filename = Path(filename)

        LOGGER.debug(f'opening "{"/".join(filename.parts[-2:])}"')

        if names is None:
            for subclass in FieldOutput.__subclasses__():
                if filename.name == subclass.filename:
                    names = subclass.variables
                    break
            else:
                raise NotImplementedError(
                    f'ADCIRC output file "{filename.name}" not implemented'
                )

        dataset = xarray.open_dataset(filename, drop_variables=cls.drop_variables)

        coordinate_variables = ['x', 'y']
        if 'depth' in dataset.variables:
            coordinate_variables += ['depth']
        coordinates = numpy.stack(
            [dataset[variable] for variable in coordinate_variables], axis=1
        )

        if filename.name in ['fort.63.nc', 'fort.64.nc']:
            data = dataset[names]
        elif filename.name in ['fort.61.nc', 'fort.62.nc']:
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
            variables_data = {}
            for variable_name in names:
                data_array = dataset[variable_name]
                if 'time_of' in variable_name:
                    pass
                if data_array.size > 0:
                    variables_data[variable_name] = numpy.squeeze(data_array)
                else:
                    LOGGER.debug(
                        f'array "{data_array.name}" has invalid data shape "{data_array.shape}"'
                    )
                    variables_data[variable_name] = numpy.squeeze(
                        numpy.full(
                            [
                                dimension if dimension > 0 else 1
                                for dimension in data_array.shape
                            ],
                            fill_value=cls.nodata,
                        )
                    )
            columns = dict(zip(coordinate_variables, coordinates.T))
            columns.update(variables_data)
            data = GeoDataFrame(
                columns, geometry=geopandas.points_from_xy(columns['x'], columns['y'])
            )
            data[data == cls.nodata] = numpy.nan

        LOGGER.debug(f'finished reading "{"/".join(filename.parts[-2:])}"')

        return data

    @classmethod
    def subset(
        cls,
        dataset: Dataset,
        bounds: (float, float, float, float) = None,
        maximum_depth: float = None,
        **kwargs,
    ) -> Dataset:
        subset = ~dataset['node'].isnull()

        if bounds is not None:
            LOGGER.debug(f'filtering within bounds {bounds}')
            if bounds[0] is not None:
                subset = xarray.ufuncs.logical_and(dataset['x'] > bounds[0])
            if bounds[2] is not None:
                subset = xarray.ufuncs.logical_and(dataset['x'] < bounds[2])
            if bounds[1] is not None:
                subset = xarray.ufuncs.logical_and(dataset['y'] > bounds[1])
            if bounds[3] is not None:
                subset = xarray.ufuncs.logical_and(dataset['y'] < bounds[3])

        if maximum_depth is not None:
            LOGGER.debug(f'filtering by maximum depth {maximum_depth}')
            subset = xarray.ufuncs.logical_and(subset, -dataset['depth'] < maximum_depth)

        return subset


class MaximumElevationOutput(FieldOutput):
    """ Maximum Elevation at All Nodes in the Model Grid (maxele.63) """

    filename = 'maxele.63.nc'
    variables = ['zeta_max', 'time_of_zeta_max']


class MaximumVelocityOutput(FieldOutput):
    """ Maximum Speed at All Nodes in the Model Grid (maxvel.63) """

    filename = 'maxvel.63.nc'
    variables = ['vel_max', 'time_of_vel_max']


class MinimumSurfacePressureOutput(FieldOutput):
    """ Minimum Sea-level Pressure at All Nodes in the Model Grid (minpr.63) """

    filename = 'minpr.63.nc'
    variables = ['pressure_min', 'time_of_pressure_min']


class MaximumSurfaceWindOutput(FieldOutput):
    """ Maximum Surface Wind Speed at All Nodes in the Model Grid (maxwvel.63) """

    filename = 'maxwvel.63.nc'
    variables = ['wind_max', 'time_of_wind_max']


class MaximumSurfaceRadiationStressOutput(FieldOutput):
    """ Maximum Radiation Surface Stress at All Nodes in the Model Grid (maxrs.63) """

    filename = 'maxrs.63.nc'
    variables = ['radstress_max', 'time_of_radstress_max']


class HotStartOutput(FieldOutput):
    """ Hot Start Output (fort.67) """

    filename = 'fort.67.nc'
    variables = ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel']


class HotStartOutput2(HotStartOutput):
    """ Hot Start Output (fort.68) """

    filename = 'fort.68.nc'


class FieldTimeSeriesOutput(FieldOutput, TimeSeriesOutput, ABC):
    pass


class ElevationTimeSeriesOutput(FieldTimeSeriesOutput):
    """ Depth-averaged Velocity Time Series at Specified Velocity Recording Stations (fort.62) """

    filename = 'fort.63.nc'
    variables = ['zeta']

    @classmethod
    def subset(
        cls,
        dataset: Dataset,
        bounds: (float, float, float, float) = None,
        maximum_depth: float = None,
        only_inundated: bool = False,
        **kwargs,
    ) -> Dataset:
        subset = super().subset(dataset, maximum_depth=maximum_depth)

        if only_inundated:
            dry_subset = dataset['zeta'].isnull()

            # get all nodes that experienced inundation (were both wet and dry at any time)
            inundated_subset = dry_subset.any('time') & ~dry_subset.all('time')

            if 'run' in dataset:
                inundated_subset = inundated_subset.any('run')

            subset = xarray.ufuncs.logical_and(subset, inundated_subset)

        return subset


class VelocityTimeSeriesOutput(FieldTimeSeriesOutput):
    """ Depth-averaged Velocity Time Series at All Nodes in the Model Grid (fort.64) """

    filename = 'fort.64.nc'
    variables = ['u-vel', 'v-vel']


class SurfacePressureTimeSeriesOutput(FieldTimeSeriesOutput):
    """ Sea-level Pressure Time Series at All Nodes in the Model Grid (fort.73) """

    filename = 'fort.73.nc'
    variables = ['pressure']


class SurfaceWindTimeSeriesOutput(FieldTimeSeriesOutput):
    """ Surface Wind Velocity Time Series at All Nodes in the Model Grid (fort.74) """

    filename = 'fort.74.nc'
    variables = ['windx', 'windy']


def adcirc_file_data_variables(cls: type = None) -> {str: [str]}:
    file_data_variables = {}
    if cls is None:
        cls = AdcircOutput
    for subclass in cls.__subclasses__():
        try:
            file_data_variables[subclass.filename] = subclass
        except AttributeError:
            file_data_variables.update(adcirc_file_data_variables(subclass))
    return file_data_variables


ADCIRC_FILE_OUTPUTS = adcirc_file_data_variables()


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
    directory: PathLike = None, file_outputs: [str] = None, parallel: bool = False,
) -> {str: dict}:
    """
    Parse output from multiple ADCIRC runs.

    :param directory: directory containing run output directories
    :param file_outputs: output files to parse
    :param parallel: load data concurrently with Dask
    :return: variables to parsed data
    """

    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    if file_outputs is None:
        file_outputs = ADCIRC_FILE_OUTPUTS
    elif isinstance(file_outputs, Collection):
        file_outputs = {filename: ADCIRC_FILE_OUTPUTS[filename] for filename in file_outputs}
    elif isinstance(file_outputs, Mapping):
        file_outputs = {
            filename: subclass if subclass is not None else ADCIRC_FILE_OUTPUTS[filename]
            for filename, subclass in file_outputs.items()
        }

    for basename, output_class in file_outputs.items():
        if isinstance(output_class, AdcircOutput):
            file_outputs[basename] = output_class.__class__

    output_tree = {}
    for basename, output_class in file_outputs.items():
        output_tree[basename] = output_class.read_directory(
            directory, variables=output_class.variables, parallel=parallel,
        )

    return output_tree


def combine_outputs(
    directory: PathLike = None,
    file_data_variables: {str: [str]} = None,
    bounds: (float, float, float, float) = None,
    maximum_depth: float = None,
    only_inundated: bool = False,
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
    if not output_filename.parent.exists():
        output_filename.mkdir(parents=True, exist_ok=True)

    runs_directory = directory / 'runs'
    if not runs_directory.exists():
        raise FileNotFoundError(f'runs directory does not exist at "{runs_directory}"')

    track_directory = directory / 'track_files'
    if not track_directory.exists():
        raise FileNotFoundError(f'track directory does not exist at "{track_directory}"')

    if file_data_variables is None:
        file_data_variables = ADCIRC_FILE_OUTPUTS
    elif isinstance(file_data_variables, Collection):
        file_data_variables = {
            filename: ADCIRC_FILE_OUTPUTS[filename] for filename in file_data_variables
        }
    elif isinstance(file_data_variables, Mapping):
        file_data_variables = {
            filename: variables if variables is not None else ADCIRC_FILE_OUTPUTS[filename]
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
        directory=runs_directory, file_outputs=file_data_variables, parallel=parallel,
    )

    if len(output_data) > 0:
        first_dataset = list(output_data.values())[0]
        runs_string = ', '.join(f'"{run}"' for run in first_dataset['run'].values)
        LOGGER.info(f'found {len(first_dataset["run"])} run(s): {runs_string}')
    else:
        LOGGER.warning(f'could not find any output files in "{directory}"')

    # generate subset
    for basename, file_data in output_data.items():
        num_nodes = len(file_data['node'])

        variable_shape_string = ', '.join(
            f'"{name}" {variable.shape}' for name, variable in file_data.items()
        )
        LOGGER.info(
            f'found {len(file_data)} variable(s) in "{basename}": {variable_shape_string}'
        )

        file_data_variable = file_data_variables[basename]

        subset = file_data_variable.subset(
            file_data,
            bounds=bounds,
            maximum_depth=maximum_depth,
            only_inundated=only_inundated,
        )

        if subset is not None:
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                file_data = file_data.sel(node=subset)

            if only_inundated and issubclass(file_data_variable, ElevationTimeSeriesOutput):
                LOGGER.info(
                    f'found {len(file_data["node"])} inundated nodes ({len(file_data["node"]) / num_nodes:3.2%} of total)'
                )

        output_data[basename] = file_data

    for basename, file_data in output_data.items():
        if output_filename is not None:
            output_netcdf_filename = output_filename.parent / basename
            LOGGER.info(f'writing to "{output_netcdf_filename}"')
            file_data.to_netcdf(
                output_netcdf_filename,
                encoding={
                    variable_name: {'zlib': True} for variable_name in file_data.variables
                },
            )

    output_data['perturbations'] = vortex_perturbations

    return output_data
