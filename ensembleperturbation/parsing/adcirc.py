from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
import pickle
from typing import Any, Collection, Dict, List, Mapping, Tuple, Union

import dask
import geopandas
from geopandas import GeoDataFrame
import numpy
import pandas
from pandas import DataFrame
from pyproj.transformer import Transformer
from scipy.spatial import KDTree
from shapely.geometry import Point
from stormevents.nhc import VortexTrack
from typepigeon import convert_value
import xarray
from xarray import DataArray, Dataset

from ensembleperturbation.perturbation.atcf import parse_vortex_perturbations
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing.adcirc')


class ElevationSelection(Enum):
    wet = 'wet'
    inundated = 'inundated'
    dry = 'dry'


class AdcircOutput(ABC):
    filename: PathLike
    variables: List[str]
    drop_variables: List[str] = ['neta', 'nvel', 'max_nvdll', 'max_nvell']
    nodata: float = -99999.0

    @classmethod
    @abstractmethod
    def read(cls, filename: PathLike, names: List[str] = None) -> Union[DataFrame, DataArray]:
        raise NotImplementedError

    @classmethod
    def async_read(
        cls, filename: PathLike, pickle_filename: PathLike, variables: List[str] = None
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
        cls, directory: PathLike, variables: List[str] = None, parallel: bool = False
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

        filename_pattern = f'*/{cls.filename}'
        filenames = list(directory.glob(filename_pattern))
        if len(filenames) > 0:
            LOGGER.info(
                f'found {len(filenames)} files matching "{directory / filename_pattern}"'
            )
        else:
            raise FileNotFoundError(
                f'could not find any files matching "{directory / filename_pattern}"'
            )

        drop_variables = cls.drop_variables

        with xarray.open_dataset(
            filenames[0], drop_variables=drop_variables
        ) as sample_dataset:
            drop_variables.extend(
                variable_name
                for variable_name in sample_dataset.variables
                if variable_name not in variables
                and variable_name not in ['node', 'time', 'x', 'y', 'depth', 'element']
            )

        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            dataset = xarray.open_mfdataset(
                filenames,
                drop_variables=drop_variables,
                combine='nested',
                concat_dim=xarray.DataArray(
                    [filename.parent.name for filename in filenames], dims=['run'], name='run',
                ),
                parallel=parallel,
                lock=False,
            )

        if 'depth' in dataset:
            dataset = dataset.assign_coords(
                {'depth': dataset['depth'].isel(run=0).drop_vars('run')}
            )

        if 'element' in dataset:
            dataset = dataset.assign_coords(  # subtract one from element table
                {'element': dataset['element'].isel(run=0).drop_vars('run') - 1}
            )

        if 'node' not in dataset:
            dataset = dataset.assign_coords({'node': dataset['node']})

        if 'element' in dataset:
            return dataset[variables].assign_coords({'element': dataset['element']})
        else:
            return dataset[variables]

    @classmethod
    @abstractmethod
    def subset(
        cls,
        dataset: Union[Dataset, DataArray],
        bounds: (float, float, float, float) = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:
        raise NotImplementedError


class TimeSeriesOutput(ABC):
    pass


class StationTimeSeriesOutput(AdcircOutput, TimeSeriesOutput, ABC):
    @classmethod
    @abstractmethod
    def read(cls, filename: PathLike, names: List[str] = None) -> GeoDataFrame:
        raise NotImplementedError

    @classmethod
    def subset(
        cls,
        dataset: Union[Dataset, DataArray],
        bounds: (float, float, float, float) = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:
        subset = ~dataset['station'].isnull()

        if bounds is not None:
            LOGGER.debug(f'filtering within bounds {bounds}')
            if bounds[0] is not None:
                subset = numpy.logical_and(subset, dataset['x'] > bounds[0])
            if bounds[2] is not None:
                subset = numpy.logical_and(subset, dataset['x'] < bounds[2])
            if bounds[1] is not None:
                subset = numpy.logical_and(subset, dataset['y'] > bounds[1])
            if bounds[3] is not None:
                subset = numpy.logical_and(subset, dataset['y'] < bounds[3])

        return subset


class ElevationStationOutput(StationTimeSeriesOutput):
    """
    ``fort.61`` - Elevation Time Series at Specified Elevation Recording Stations

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/elevation-time-series-specified-elevation-recording-stations-fort-61
    """

    filename = 'fort.61.nc'
    variables = ['station_name', 'zeta']

    @classmethod
    def read(cls, filename: PathLike, names: List[str] = None) -> GeoDataFrame:
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
    """
    ``fort.62`` - Depth-averaged Velocity Time Series at Specified Velocity Recording Stations

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/depth-averaged-velocity-time-series-specified-velocity-recording-stations-fort-62
    """

    filename = 'fort.62.nc'
    variables = ['station_name', 'u-vel', 'v-vel']

    @classmethod
    def read(cls, filename: PathLike, names: List[str] = None) -> GeoDataFrame:
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
    def read(cls, filename: PathLike, names: List[str] = None) -> Union[DataFrame, DataArray]:
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
        if 'element' in dataset.variables:
            coordinate_variables += ['element']
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
        dataset: Union[Dataset, DataArray],
        bounds: (float, float, float, float) = None,
        wind_swath: [str, int] = None,
        maximum_depth: float = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:
        subset = ~dataset['node'].isnull()

        if wind_swath is not None:
            cyclone = wind_swath[0]
            isotach = wind_swath[1]
            LOGGER.debug(f'filtering within {cyclone} wind swath {isotach}')
            if not isinstance(cyclone, VortexTrack):
                try:
                    cyclone = VortexTrack.from_fort22(cyclone)
                except FileNotFoundError:
                    cyclone = VortexTrack(cyclone)
            swath = cyclone.wind_swaths(wind_speed=isotach)
            polygon = GeoDataFrame(index=[0], geometry=[next(iter(swath.values()))])
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                geometry = geopandas.points_from_xy(dataset['x'].values, dataset['y'].values)
                points = GeoDataFrame(
                    {'lon': dataset['x'].values, 'lat': dataset['y'].values},
                    geometry=geometry,
                )
                inpoly = geopandas.tools.sjoin(points, polygon, predicate='within', how='left')
            subset = numpy.logical_and(subset, ~numpy.isnan(inpoly.index_right.values))

        if bounds is not None:
            LOGGER.debug(f'filtering within bounds {bounds}')
            if bounds[0] is not None:
                subset = numpy.logical_and(subset, dataset['x'] > bounds[0])
            if bounds[2] is not None:
                subset = numpy.logical_and(subset, dataset['x'] < bounds[2])
            if bounds[1] is not None:
                subset = numpy.logical_and(subset, dataset['y'] > bounds[1])
            if bounds[3] is not None:
                subset = numpy.logical_and(subset, dataset['y'] < bounds[3])

        if maximum_depth is not None:
            LOGGER.debug(f'filtering by maximum depth {maximum_depth}')
            subset = numpy.logical_and(subset, dataset['depth'] < maximum_depth)

        return subset


class MaximumElevationOutput(FieldOutput):
    """
    ``maxele.63`` - Maximum Elevation at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/global-maximum-minimum-files-model-run-maxele-63-maxvel-63-maxwvel-63-maxrs-63-minpr-63/
    """

    filename = 'maxele.63.nc'
    variables = ['zeta_max', 'time_of_zeta_max']


class MaximumVelocityOutput(FieldOutput):
    """
    ``maxvel.63`` - Maximum Speed at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/global-maximum-minimum-files-model-run-maxele-63-maxvel-63-maxwvel-63-maxrs-63-minpr-63/
    """

    filename = 'maxvel.63.nc'
    variables = ['vel_max', 'time_of_vel_max']


class MinimumSurfacePressureOutput(FieldOutput):
    """
    ``minpr.63`` - Minimum Sea-level Pressure at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/global-maximum-minimum-files-model-run-maxele-63-maxvel-63-maxwvel-63-maxrs-63-minpr-63/
    """

    filename = 'minpr.63.nc'
    variables = ['pressure_min', 'time_of_pressure_min']


class MaximumSurfaceWindOutput(FieldOutput):
    """
    ``maxwvel.63`` - Maximum Surface Wind Speed at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/global-maximum-minimum-files-model-run-maxele-63-maxvel-63-maxwvel-63-maxrs-63-minpr-63/
    """

    filename = 'maxwvel.63.nc'
    variables = ['wind_max', 'time_of_wind_max']


class MaximumSurfaceRadiationStressOutput(FieldOutput):
    """
    ``maxrs.63`` - Maximum Radiation Surface Stress at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/global-maximum-minimum-files-model-run-maxele-63-maxvel-63-maxwvel-63-maxrs-63-minpr-63/
    """

    filename = 'maxrs.63.nc'
    variables = ['radstress_max', 'time_of_radstress_max']


class HotStartOutput(FieldOutput):
    """
    ``fort.67`` - Hot Start Output

    https://adcirc.org/home/documentation/users-manual-v52/input-file-descriptions/hot-start-files-fort-67-fort-68/
    """

    filename = 'fort.67.nc'
    variables = ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel']


class HotStartOutput2(HotStartOutput):
    """
    ``fort.68`` - Hot Start Output

    https://adcirc.org/home/documentation/users-manual-v52/input-file-descriptions/hot-start-files-fort-67-fort-68/
    """

    filename = 'fort.68.nc'


class FieldTimeSeriesOutput(FieldOutput, TimeSeriesOutput, ABC):
    pass


class ElevationTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``fort.63`` - Elevation Time Series at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/elevation-time-series-nodes-model-grid-fort-63
    """

    filename = 'fort.63.nc'
    variables = ['zeta']

    @classmethod
    def subset(
        cls,
        dataset: Union[Dataset, DataArray],
        bounds: (float, float, float, float) = None,
        maximum_depth: float = None,
        elevation_selection: ElevationSelection = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:
        subset = super().subset(dataset, bounds=bounds, maximum_depth=maximum_depth)

        if elevation_selection is not None:
            if not isinstance(elevation_selection, ElevationSelection):
                elevation_selection = convert_value(elevation_selection, ElevationSelection)

            dry_subset = dataset['zeta'].isnull()

            if elevation_selection == ElevationSelection.wet:
                elevation_subset = ~dry_subset.any('time')
            elif elevation_selection == ElevationSelection.inundated:
                # get all nodes that experienced inundation (were both wet and dry at any time)
                elevation_subset = dry_subset.any('time') & ~dry_subset.all('time')
            else:
                elevation_subset = dry_subset.all('time')

            if 'run' in dataset:
                elevation_subset = elevation_subset.any('run')

            subset = numpy.logical_and(subset, elevation_subset)

        return subset


class VelocityTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``fort.64`` - Depth-averaged Velocity Time Series at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/depth-averaged-velocity-time-series-nodes-model-grid-fort-64
    """

    filename = 'fort.64.nc'
    variables = ['u-vel', 'v-vel']


class SurfacePressureTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``fort.73`` - Sea-level Pressure Time Series at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/output-file-descriptions/atmospheric-pressure-time-series-nodes-model-grid-fort-73
    """

    filename = 'fort.73.nc'
    variables = ['pressure']


class SurfaceWindTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``fort.74`` - Surface Wind Velocity Time Series at All Nodes in the Model Grid

    https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions/wind-stress-velocity-time-series-nodes-model-grid-fort-74
    """

    filename = 'fort.74.nc'
    variables = ['windx', 'windy']


def adcirc_file_data_variables(cls: type = None) -> Dict[str, List[str]]:
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
    directory: PathLike = None, file_outputs: List[str] = None, parallel: bool = False,
) -> Dict[str, dict]:
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
        if issubclass(output_class, AdcircOutput):
            file_outputs[basename] = output_class

    output_tree = {}
    for basename, output_class in file_outputs.items():
        try:
            output_tree[basename] = output_class.read_directory(
                directory, variables=output_class.variables, parallel=parallel,
            )
        except (ValueError, FileNotFoundError) as error:
            LOGGER.warning(error)

    return output_tree


def combine_outputs(
    directory: PathLike = None,
    file_data_variables: Dict[str, List[str]] = None,
    bounds: Tuple[float, float, float, float] = None,
    maximum_depth: float = None,
    elevation_selection: ElevationSelection = None,
    output_directory: PathLike = None,
    parallel: bool = False,
) -> Dict[str, DataFrame]:
    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    if maximum_depth is not None and not isinstance(maximum_depth, float):
        maximum_depth = float(maximum_depth)

    if output_directory is not None:
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)

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
    output_data = {'perturbations.nc': parse_vortex_perturbations(track_directory)}

    # parse all the outputs using built-in parser
    LOGGER.info(f'parsing from "{directory}"')
    parsed_files = parse_adcirc_outputs(
        directory=runs_directory, file_outputs=file_data_variables, parallel=parallel,
    )

    elevation_time_series_filename = 'fort.63.nc'
    if elevation_selection is not None:
        if elevation_time_series_filename in parsed_files:
            output_data[elevation_time_series_filename] = parsed_files[
                elevation_time_series_filename
            ]
            del parsed_files[elevation_time_series_filename]
        else:
            raise ValueError(
                f'elevation time series "{elevation_time_series_filename}" not found'
            )

    output_data.update(parsed_files)

    # generate subset
    elevation_subset = None
    for basename, file_data in output_data.items():
        if 'node' in file_data:
            num_nodes = len(file_data['node'])

            variable_shape_string = ', '.join(
                f'"{name}" {variable.shape}' for name, variable in file_data.items()
            )
            LOGGER.info(
                f'found {len(file_data)} variable(s) in "{basename}": {variable_shape_string}'
            )

            file_data_variable = file_data_variables[basename]

            subset = ~file_data['node'].isnull()

            if elevation_subset is not None:
                subset = numpy.logical_and(subset, elevation_subset)

            subset = numpy.logical_and(
                subset,
                file_data_variable.subset(
                    file_data,
                    bounds=bounds,
                    maximum_depth=maximum_depth,
                    elevation_selection=elevation_selection,
                ),
            )

            if subset is not None:
                file_data = file_data.sel(node=subset)

                LOGGER.info(
                    f'subsetted {len(file_data["node"])} out of {num_nodes} total nodes ({len(file_data["node"]) / num_nodes:3.2%})'
                )

                if elevation_selection is not None:
                    elevation_subset = subset

            output_data[basename] = file_data

    for basename, file_data in output_data.items():
        if output_directory is not None:
            output_filename = output_directory / basename
            LOGGER.info(f'writing to "{output_filename}"')
            file_data.to_netcdf(
                output_filename,
                # encoding={
                #     variable_name: {'zlib': True} for variable_name in file_data.variables
                # },
            )

    return output_data


def subset_dataset(
    ds: Dataset,
    variable: str,
    maximum_depth: float = None,
    wind_swath: list = None,
    bounds: (float, float, float, float) = None,
    node_status_selection: dict = None,
    point_spacing: int = None,
    output_filename: PathLike = None,
):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        if node_status_selection is None:
            node_status_mask = ~ds[variable].isnull()
        elif node_status_selection['mask'] == 'sometimes_wet':
            node_status_mask = (
                ~ds[variable].sel(run=node_status_selection['runs']).isnull().all('run'),
            )
        elif node_status_selection['mask'] == 'always_wet':
            node_status_mask = (
                ~ds[variable].sel(run=node_status_selection['runs']).isnull().any('run'),
            )
        else:
            raise f'node_status_selection {node_status_selection["mask"]} unrecognized'

        node_subset_mask = (
            FieldOutput.subset(
                ds['node'], maximum_depth=maximum_depth, bounds=bounds, wind_swath=wind_swath,
            ),
        )
        subsetted_nodes = ds['node'].values[
            numpy.logical_and(node_status_mask, node_subset_mask).squeeze()
        ]
        if point_spacing is not None:
            subsetted_nodes = subsetted_nodes[::point_spacing]
        subset = ds.sel(node=subsetted_nodes)

        # adjust element table if present
        if 'element' in subset:
            # keep only elements where all nodes are present
            elements = subset['element'].values
            element_mask = numpy.isin(elements, subset['node'].values).all(axis=1)
            elements = elements[element_mask]
            # map nodes in element table to local numbering system (start at 0)
            node_mapper = numpy.zeros(subset['node'].max().values + 1, dtype=int)
            node_index = numpy.arange(len(subset['node']))
            node_mapper[subset['node'].values] = node_index
            elements = node_mapper[elements]
            # update element table in dataset
            ele_da = DataArray(data=elements, dims=['nele', 'nvertex'])
            subset = subset.assign_coords({'element': ele_da})
        try:
            subset = subset.drop_sel(run='original')
        except:
            pass
        if len(subset['node']) != len(ds['node']):
            LOGGER.info(
                f'subsetted down to {len(subset["node"])} nodes ({len(subset["node"]) / len(ds["node"]):.1%})'
            )
        if output_filename is not None:
            if not isinstance(output_filename, Path):
                output_filename = Path(output_filename)
            LOGGER.info(f'saving subset to "{output_filename}"')
            subset.to_netcdf(output_filename)

    return subset


def extrapolate_water_elevation_to_dry_areas(
    da: DataArray,
    k_neighbors: int = 1,
    idw_order: int = 1,
    compute_headloss: bool = False,
    mann_coef: float = 0.05,
    u_ref: float = 0.4,
    d_ref: float = 1.0,
):
    # return a deep copy of original datarrary on output
    da_adjusted = da.copy(deep=True)

    # Get coordinates in conformal projection (e.g,, Mercator)
    # for determining closest distance
    crs_from = 'EPSG:4326'  # WGS84
    crs_to = 'EPSG:3857'  # Mercator
    transformer = Transformer.from_crs(crs_from=crs_from, crs_to=crs_to, always_xy=True)
    x, y = transformer.transform(da['x'].values, da['y'].values)
    projected_coordinates = numpy.vstack([x, y]).T

    # for mapping back to node numbers
    nodes = numpy.arange(da.sizes['node'])

    # compute the friction factor for headloss calculation:
    # Rucker, et al. (2021). Natural Hazards.
    # https://doi.org/10.1007/s11069-021-04634-8
    if compute_headloss:
        friction_factor = (u_ref * mann_coef) ** 2 / d_ref ** (4 / 3)
    else:
        friction_factor = 0.0

    # inverse distance weighting of order `idw_order` with `k_nearest` neighbors
    for run in range(da.sizes['run']):
        null = numpy.isnan(da[run, :])
        tree = KDTree(projected_coordinates[~null])
        dd, nn = tree.query(projected_coordinates[null], k=k_neighbors)
        if k_neighbors == 1:
            headloss = dd * friction_factor  # hydraulic friction loss
            da_adjusted[run, null] = da[run, nodes[~null][nn]].values - headloss
        else:
            for kk in range(k_neighbors):
                weights = dd[:, kk] ** (-idw_order)
                headloss = dd[:, kk] * friction_factor  # hydraulic friction loss
                total_head = da[run, nodes[~null][nn[:, kk]]].values - headloss
                if kk == 0:
                    idw_sum = total_head * weights
                    weight_sum = weights
                else:
                    idw_sum += total_head * weights
                    weight_sum += weights
            da_adjusted[run, null] = idw_sum / weight_sum

    return da_adjusted
