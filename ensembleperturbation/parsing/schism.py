from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Tuple, Union
from functools import reduce
from copy import deepcopy
from collections import UserDict
import fnmatch

import dask
import geopandas
import f90nml
from geopandas import GeoDataFrame
import numpy
import pandas
from pandas import DataFrame
from pyproj.transformer import Transformer
from pyschism.mesh import Hgrid
from scipy.spatial import KDTree
from shapely.geometry import Point
from stormevents.nhc import VortexTrack
from typepigeon import convert_value
import xarray
from xarray import DataArray, Dataset

from ensembleperturbation.perturbation.atcf import parse_vortex_perturbations
from ensembleperturbation.utilities import get_logger


LOGGER = get_logger('parsing.schism')

SCHISM_ADCIRC_COORD_MAPPING = {
    'nSCHISM_hgrid_node': 'node',
    'SCHISM_hgrid_node_x': 'x',
    'SCHISM_hgrid_node_y': 'y',
}

SCHISM_ADCIRC_VAR_MAPPING = {
    'elevation': 'zeta',
    'horizontalVelX': 'u-vel',
    'horizontalVelY': 'v-vel',
    'windSpeedX': 'windx',
    'windSpeedY': 'windy',
    'airPressure': 'pressure',
    'max_elevation': 'zeta_max',
    'max_elevation_times': 'time_of_zeta_max',
    'max_velocity': 'vel_max',
    'max_velocity_times': 'time_of_vel_max',
    'min_pressure': 'pressure_min',
    'min_pressure_times': 'time_of_pressure_min',
    'max_wind': 'wind_max',
    'max_wind_times': 'time_of_wind_max',
    'station_index': 'station_name',
}

SCHISM_ADCIRC_OUT_MAPPING = {
    'schism_point_elevtion.nc': 'fort.61.nc',  # ['station_name', 'zeta']
    'schism_point_velocity.nc': 'fort.62.nc',  # ['station_name', 'u-vel', 'v-vel']
    'schism_max_elevation.nc': 'maxele.63.nc',  # ['zeta_max', 'time_of_zeta_max']
    'schism_max_velocity.nc': 'maxvel.63.nc',  # ['vel_max', 'time_of_vel_max']
    'schism_min_pressure.nc': 'minpr.63.nc',  # ['pressure_min', 'time_of_pressure_min']
    'schism_max_wind.nc': 'maxwvel.63.nc',  # ['wind_max', 'time_of_wind_max']
    'schism_elevation.nc': 'fort.63.nc',  # ['zeta']
    'schism_velocity.nc': 'fort.64.nc',  # ['u-vel', 'v-vel']
    'schism_pressure.nc': 'fort.73.nc',  # ['pressure']
    'schism_wind.nc': 'fort.74.nc',  # ['windx', 'windy']
}


def is_stacked(pattern):
    return '*' in pattern


def create_output_dict(file_pattern, directory=Path(), existing_dict=None):
    output_dict = {}
    if isinstance(existing_dict, Mapping):
        output_dict = deepcopy(existing_dict)

    matches = list(directory.glob(f'**/{file_pattern}'))
    proper_matches = [path for path in matches if path.parent.name == 'outputs']
    run_dirs = set(path.parent.parent for path in proper_matches)
    for rundir in run_dirs:
        run_outputs = [path for path in proper_matches if path.match(f'{rundir}/outputs/*')]
        n_stacks = 1
        if is_stacked(file_pattern):
            n_stacks = max(int(out.stem.split('_')[-1]) for out in run_outputs)
        run_dict = output_dict.setdefault(rundir, dict())
        run_dict['name'] = rundir.name
        run_out_dict = run_dict.setdefault('outputs', dict())
        pattern_dict = run_out_dict.setdefault(file_pattern, dict())
        pattern_dict['n_stacks'] = n_stacks
        pattern_dict['files'] = run_outputs

    return output_dict


def validate_run_output(run_dict, output_patterns):
    # Check if it has all patterns
    if not all(patt in run_dict['outputs'] for patt in output_patterns):
        return False

    # Check if num of stacks match, for stacked (in time) outputs
    if (
        len(
            {
                run_dict['outputs'][patt]['n_stacks']
                for patt in output_patterns
                if is_stacked(patt)
            }
        )
        > 1
    ):
        return False

    return True


def find_run_dir_for_output(file_patterns, directory=Path(), validate=True):

    output_dict = {}
    for pattern in file_patterns:
        output_dict = create_output_dict(pattern, directory, output_dict)

    if validate:
        validated_outputs = {}
        for key, run_dict in output_dict.items():
            if validate_run_output(run_dict, file_patterns):
                validated_outputs[key] = run_dict
        output_dict = validated_outputs

    if len(output_dict) > 0:
        LOGGER.info(
            f'found {len(output_dict)} run directories with all the specified output patterns'
        )
    else:
        raise FileNotFoundError(
            f'could not find any run directories with all the specified output patterns'
        )

    return output_dict


class ElevationSelection(Enum):
    wet = 'wet'
    inundated = 'inundated'
    dry = 'dry'


class SchismOutput(ABC):
    out_filename: str
    file_patterns: str
    variables: List[str]
    drop_variables: List[str] = []
    nodata: float = -99999.0

    @classmethod
    @abstractmethod
    def read(cls, filename: PathLike, names: List[str] = None) -> Union[DataFrame, DataArray]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def read_directory(
        cls, directory: PathLike, variables: List[str] = None, parallel: bool = False
    ) -> Dataset:
        raise NotImplementedError

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


class StationTimeSeriesOutput(SchismOutput, TimeSeriesOutput, ABC):

    station_file = 'station.in'

    """
    outputs/staout_[1..,9], corresponding respectively to elev,
    air pressure, wind u, wind v, T, S, u, v, w
    """

    @classmethod
    def read(
        cls,
        filenames: Union[PathLike, List[PathLike]],
        station_file: PathLike,
        station_idx: List[int] = None,
    ) -> GeoDataFrame:

        if not isinstance(filenames, List):
            filenames = [filenames]

        if not all(any(j.match(i) for j in filenames) for i in cls.file_patterns):
            raise ValueError(
                f'{cls.__name__} requires files that match {" & ".join(cls.file_patterns)}'
                f' input files for {" & ".join(cls.variables)}'
            )

        station_df = (
            pandas.read_csv(
                station_file, header=None, delim_whitespace=True, index_col=0, skiprows=2
            )
            .drop(columns=[3, 4])
            .rename_axis('station_index')
            .rename(columns={1: 'x', 2: 'y'})
        )

        all_results_df = pandas.DataFrame()
        for fname, var in zip(filenames, cls.variables):
            result_df = (
                pandas.read_csv(fname, header=None, delim_whitespace=True, index_col=0)
                .transpose()
                .melt(var_name='time', ignore_index=False)
                .rename_axis('station_index')
                .rename(columns={'value': var})
            )
            result_df['x'] = station_df.loc[
                result_df.index.get_level_values('station_index')
            ].x
            result_df['y'] = station_df.loc[
                result_df.index.get_level_values('station_index')
            ].y
            result_df = result_df.set_index(['time', result_df.index])
            all_results_df = pandas.concat((all_results_df, result_df), axis=1)
        all_results_df = all_results_df.reset_index()

        # TODO: Use station_idx if none to only return stations of interest
        station_gdf = geopandas.GeoDataFrame(
            geometry=geopandas.points_from_xy(
                station_df.loc[all_results_df.station_index].x,
                station_df.loc[all_results_df.station_index].y,
            ),
            data=all_results_df,
        )

        return station_gdf

    @classmethod
    def read_to_dataset(
        cls,
        filenames: Union[PathLike, List[PathLike]],
        station_file: PathLike,
        station_idx: List[int] = None,
        run_name: str = None,
    ) -> Dataset:

        station_gdf = cls.read(filenames, station_file, station_idx)
        if run_name is not None:
            station_gdf['run'] = run_name

        dataset = xarray.Dataset.from_dataframe(
            station_gdf.set_index(['run', 'time', 'station_index'])
        )
        dataset = dataset.assign(
            {
                'geometry': dataset.geometry.isel(run=0, time=0),
                'x': dataset.x.isel(run=0, time=0),
                'y': dataset.y.isel(run=0, time=0),
            }
        )

        return dataset

    @classmethod
    def read_directory(
        cls, directory: PathLike, variables: List[str] = None, parallel: bool = False
    ) -> GeoDataFrame:
        """
        Compile a dataset from output files in the given directory.
        This could be called for a single run directory or a parent
        of multiple run directories.

        :param directory: directory containing output files
        :param parallel: load data concurrently with Dask
        :return: GeoDataFrame of output data
        """

        if not isinstance(directory, Path):
            directory = Path(directory)

        output_dict = find_run_dir_for_output(cls.file_patterns, directory)
        run_dirs = list(output_dict.keys())

        # TODO: Use variables?

        dataset_list = []
        lazy_results = []
        for en, run_dir in enumerate(run_dirs):
            if parallel:
                lazy_results.append(
                    dask.delayed(cls.read_to_dataset)(
                        [
                            f
                            for patt_dict in output_dict[run_dir]['outputs'].values()
                            for f in patt_dict['files']
                        ],
                        run_dir / cls.station_file,
                        run_name=run_dir.name,
                    )
                )

            else:
                dataset_list.append(
                    cls.read_to_dataset(
                        [
                            f
                            for patt_dict in output_dict[run_dir]['outputs'].values()
                            for f in patt_dict['files']
                        ],
                        run_dir / cls.station_file,
                        run_name=run_dir.name,
                    )
                )

        if len(lazy_results) > 0:
            dataset_list = dask.compute(*lazy_results)

        return xarray.merge(dataset_list)

    @classmethod
    def subset(
        cls,
        dataset: Union[Dataset, DataArray],
        bounds: (float, float, float, float) = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:

        subset = ~dataset['station_index'].isnull()

        if bounds is not None:
            LOGGER.debug(f'filtering within bounds {bounds}')
            if bounds[0] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_x'] > bounds[0])
            if bounds[2] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_x'] < bounds[2])
            if bounds[1] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_y'] > bounds[1])
            if bounds[3] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_y'] < bounds[3])

        return subset


class ElevationStationOutput(StationTimeSeriesOutput):
    """
    ``staout_1`` - Elevation Time Series at Specified Elevation Recording Stations

    """

    out_filename = 'schism_point_elevtion.nc'
    file_patterns = ['staout_1']
    variables = ['elevation']


class VelocityStationOutput(StationTimeSeriesOutput):
    """
    ``staout_7`` & ``staout_8`` - Depth-averaged Velocity Time Series
    at Specified Velocity Recording Stations

    """

    out_filename = 'schism_point_velocity.nc'
    file_patterns = ['staout_7', 'staout_8']
    variables = ['horizontalVelX', 'horizontalVelY']


class FieldOutput(SchismOutput, ABC):
    @classmethod
    def read(
        cls, filenames: Union[PathLike, List[PathLike]], names: List[str] = None
    ) -> Union[DataFrame, DataArray]:
        """
        Parse SCHISM output files

        :param filenames: file path to SCHISM NetCDF outputs
        :param names: list of data variables to extract
        :return: parsed data
        """

        if not isinstance(filenames, List):
            filenames = [filenames]

        if all(any(j.matches(i) for j in filenames) for i in cls.file_patterns):
            raise ValueError(
                f'{cls.__name__} requires files that match {" & ".join(cls.file_patterns)}'
                f' input files for {" & ".join(cls.variables)}'
            )

        filenames = [Path(fnm) for fnm in filenames]

        for filename in filenames:
            LOGGER.debug(f'opening "{"/".join(filename.parts[-2:])}"')

        if names is None:
            names = []
            for subclass in FieldOutput.__subclasses__():
                # NOTE: Same filename can contain info for multiple vars (out2d)
                if all(
                    [
                        fpath.name == Path(predef).name
                        for fpath, predef in zip(filenames, subclass.filenames)
                    ]
                ):
                    names.extend(subclass.variables)
            else:
                raise NotImplementedError(
                    f'Support for one of the provided SCHISM output files is not implemented'
                )
        names = list(set(names))

        dataset = xarray.open_mfdataset(filenames, drop_variables=cls.drop_variables)
        data = dataset[names]
        data = data.assign_attrs(**dataset.attrs)

        for filename in filenames:
            LOGGER.debug(f'finished reading "{"/".join(filename.parts[-2:])}"')

        return data

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

        output_dict = find_run_dir_for_output(cls.file_patterns, directory)

        file_collection = [
            [[f for f in patt_dict['files']] for patt_dict in run_dict['outputs'].values()]
            for run_dict in output_dict.values()
        ]
        run_dirs = list(output_dict.keys())

        # Open only the first set of outputs to get all drop_variables
        drop_variables = deepcopy(cls.drop_variables)

        with xarray.open_mfdataset(
            [flist[0] for flist in file_collection[0]], drop_variables=drop_variables
        ) as sample_dataset:
            drop_variables.extend(
                variable_name
                for variable_name in sample_dataset.variables
                if variable_name not in variables
                and variable_name
                not in [
                    'time',
                    'SCHISM_hgrid_node_x',
                    'SCHISM_hgrid_node_y',
                    'depth',
                    'dryFlagNode',
                    'minimum_depth',
                ]
            )
            if not all(var in sample_dataset.data_vars for var in variables):
                LOGGER.warn("Files don't contain all the required variables!")
                return xarray.Dataset()

        # Now open all the relevant output files for all runs
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            dataset = xarray.open_mfdataset(
                [f for flist in file_collection[0] for f in flist],
                drop_variables=drop_variables,
                parallel=parallel,
                lock=False,
            ).expand_dims({'run': [run_dirs[0].name]})
            for run_idx, run_dir in enumerate(run_dirs[1:]):
                run_out_filelist = [f for flist in file_collection[run_idx + 1] for f in flist]
                dataset = xarray.combine_nested(
                    [
                        dataset,
                        xarray.open_mfdataset(
                            run_out_filelist,
                            drop_variables=drop_variables,
                            parallel=parallel,
                            lock=False,
                        ).expand_dims({'run': [run_dir.name]}),
                    ],
                    concat_dim='run',
                )

        # Drop run dimension for variables fixed across runs
        # `SCHISM_hgrid_node_x` and `SCHISM_hgrid_node_y` are
        # coordinates and are not expanded in `run` dimension
        fixed_vars = [
            'node',
            'depth',
        ]
        for var in fixed_vars:
            if var not in dataset:
                continue
            dataset = dataset.assign({var: dataset[var].isel(run=0)})

        # Add element table
        dataset = cls._add_element_table(dataset, directory)

        # TODO: What if it's different for different runs?
        if 'minimum_depth' in dataset:
            dataset = dataset.assign_attrs(
                minimum_depth=dataset.minimum_depth.isel(one=0, run=0).values
            )

        # TODO: Does it make sense to have all these as "coord" so that
        # we can return a DataArray?!
        coord_vars = [
            'SCHISM_hgrid_node_x',
            'SCHISM_hgrid_node_y',
            'depth',
            'element',
            'node',
            'dryFlagNode',
        ]
        for var in coord_vars:
            if var in dataset:
                dataset = dataset.assign_coords({var: dataset[var]})

        ret_value = dataset[variables]
        if 'element' in dataset and 'element' not in ret_value:
            ret_value = ret_value.assign_coords({'element': dataset['element']})
        ret_value = ret_value.assign_attrs(**dataset.attrs)
        return ret_value

    @classmethod
    def subset(
        cls,
        dataset: Union[Dataset, DataArray],
        bounds: (float, float, float, float) = None,
        wind_swath: [str, int] = None,
        maximum_depth: float = None,
        minimum_depth: float = None,
        **kwargs,
    ) -> Union[Dataset, DataArray]:
        # TODO: Is this right?
        subset = ~dataset['nSCHISM_hgrid_node'].isnull()

        if wind_swath is not None:
            cyclone = wind_swath[0]
            isotach = wind_swath[1]
            LOGGER.debug(f'filtering within {cyclone} wind swath {isotach}')
            if not isinstance(cyclone, VortexTrack):
                try:
                    cyclone = VortexTrack.from_file(cyclone)
                except FileNotFoundError:
                    cyclone = VortexTrack(cyclone)
            swath = cyclone.wind_swaths(wind_speed=isotach)
            if 'BEST' in swath:
                tracks = swath['BEST']
            elif 'OFCL' in swath:
                tracks = swath['OFCL']
            else:
                raise ValueError(
                    'Neither best or official track could be found for the specified storm'
                )

            series = pandas.Series(tracks.keys())
            latest_track = tracks[series[pandas.to_datetime(series).argmax()]]

            polygon = GeoDataFrame(index=[0], geometry=[latest_track])
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                geometry = geopandas.points_from_xy(
                    dataset['SCHISM_hgrid_node_x'].values,
                    dataset['SCHISM_hgrid_node_y'].values,
                )
                points = GeoDataFrame(
                    {
                        'lon': dataset['SCHISM_hgrid_node_x'].values,
                        'lat': dataset['SCHISM_hgrid_node_y'].values,
                    },
                    geometry=geometry,
                )
                inpoly = geopandas.tools.sjoin(points, polygon, predicate='within', how='left')
            subset = numpy.logical_and(subset, ~numpy.isnan(inpoly.index_right.values))

        if bounds is not None:
            LOGGER.debug(f'filtering within bounds {bounds}')
            if bounds[0] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_x'] > bounds[0])
            if bounds[2] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_x'] < bounds[2])
            if bounds[1] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_y'] > bounds[1])
            if bounds[3] is not None:
                subset = numpy.logical_and(subset, dataset['SCHISM_hgrid_node_y'] < bounds[3])

        if maximum_depth is not None:
            LOGGER.debug(f'filtering by maximum depth {maximum_depth}')
            subset = numpy.logical_and(subset, dataset['depth'] <= maximum_depth)

        if minimum_depth is not None:
            LOGGER.debug(f'filtering by minimum depth {minimum_depth}')
            subset = numpy.logical_and(subset, dataset['depth'] >= minimum_depth)

        return subset

    @classmethod
    def _add_element_table(cls, dataset: Dataset, directory: PathLike) -> Dataset:
        # hgrid.gr3 or ll
        gridfile_pattern = 'hgrid.*'
        matches = list(directory.glob(f'**/{gridfile_pattern}'))

        if len(matches) == 0:
            return dataset

        # TODO: Check if all the found hgrid files are the same
        gridfile = matches[0]

        # NOTE: All elements are treated as tria (quads are split)
        grid = Hgrid.open(gridfile, crs=4326)
        dataset = dataset.assign(
            element=xarray.DataArray(
                data=grid.elements.triangulation.triangles, dims=('nele', 'nvertex')
            )
        )

        return dataset


class ExtremumScalarFieldOutputCalculator(FieldOutput):

    derived_name: str
    derived_time_name: str
    extermum_func: str

    @classmethod
    def read(
        cls, filenames: Union[PathLike, List[PathLike]], names: List[str] = None
    ) -> Union[DataFrame, DataArray]:
        full_ds = super().read(filenames, names)
        ds = cls._calc_extermum(full_ds)
        return ds

    @classmethod
    def read_directory(
        cls, directory: PathLike, variables: List[str] = None, parallel: bool = False
    ) -> Dataset:
        if variables is None:
            variables = cls.variables
        full_ds = super().read_directory(directory, variables, parallel)
        if all(var in full_ds.data_vars for var in variables):
            ds = cls._calc_extermum(full_ds)
            return ds

        return xarray.Dataset()

    @classmethod
    def _calc_extermum(cls, full_ds) -> Dataset:
        if len(cls.variables) > 1:
            to_extrm_ary = (
                numpy.sum(getattr(full_ds, var) ** 2 for var in cls.variables) ** 0.5
            )
        else:
            to_extrm_ary = full_ds[cls.variables[0]]

        uses_dask_array = False
        if to_extrm_ary.chunks is not None and len(to_extrm_ary.chunks) > 0:
            uses_dask_array = True

        if uses_dask_array:
            # compute() due to https://github.com/pydata/xarray/issues/2511
            arg_extrm_var = getattr(to_extrm_ary, cls.extermum_func)(dim='time').compute()
        else:
            arg_extrm_var = getattr(to_extrm_ary, cls.extermum_func)(dim='time')
        extrm_vals = to_extrm_ary.isel(time=arg_extrm_var)
        extrm_times = to_extrm_ary.time.isel(time=arg_extrm_var)

        # TODO: Chunk the dataset?
        ds = xarray.Dataset(
            {
                cls.derived_name: (extrm_vals.dims, extrm_vals.data),
                cls.derived_time_name: (extrm_vals.dims, extrm_times.data),
                'run': extrm_vals.run.data,
                'nSCHISM_hgrid_node': extrm_vals.nSCHISM_hgrid_node.data,
            },
        )
        ds = ds.assign_attrs(**full_ds.attrs)
        if 'SCHISM_hgrid_node_x' in full_ds.data_vars:
            ds['SCHISM_hgrid_node_x'] = full_ds.SCHISM_hgrid_node_x
        elif 'SCHISM_hgrid_node_x' in full_ds.coords:
            ds = ds.assign_coords({'SCHISM_hgrid_node_x': full_ds.SCHISM_hgrid_node_x})
        if 'SCHISM_hgrid_node_y' in full_ds.data_vars:
            ds['SCHISM_hgrid_node_y'] = full_ds.SCHISM_hgrid_node_y
        elif 'SCHISM_hgrid_node_y' in full_ds.coords:
            ds = ds.assign_coords({'SCHISM_hgrid_node_y': full_ds.SCHISM_hgrid_node_y})

        return ds


class MaximumScalarFieldOutputCalculator(ExtremumScalarFieldOutputCalculator):
    extermum_func = 'argmax'


class MinimumScalarFieldOutputCalculator(ExtremumScalarFieldOutputCalculator):
    extermum_func = 'argmin'


class MaximumElevationOutput(MaximumScalarFieldOutputCalculator):
    """
    ``out2d.nc`` - Derived Maximum Elevation at All Nodes in the Model Grid

    """

    out_filename = 'schism_max_elevation.nc'
    file_patterns = ['out2d_*.nc']
    variables = ['elevation']
    derived_name = 'max_elevation'
    derived_time_name = 'max_elevation_times'

    @classmethod
    def read(
        cls, filenames: Union[PathLike, List[PathLike]], names: List[str] = None
    ) -> Union[DataFrame, DataArray]:
        dataset = super().read(filenames, names)
        dataset.attrs['h0'] = dataset.minimum_depth
        return cls._set_dry_to_null(dataset)

    @classmethod
    def read_directory(
        cls, directory: PathLike, variables: List[str] = None, parallel: bool = False
    ) -> Dataset:

        dataset = super().read_directory(directory, variables, parallel)
        dataset.attrs['h0'] = dataset.minimum_depth
        return cls._set_dry_to_null(dataset)

    @classmethod
    def _set_dry_to_null(cls, dataset: Dataset) -> Dataset:
        dataset[cls.derived_name] = dataset[cls.derived_name].where(
            dataset[cls.derived_name] > dataset.attrs['h0'], numpy.nan
        )

        return dataset


class MaximumVelocityOutput(MaximumScalarFieldOutputCalculator):
    """
    ``horizontalVelX.nc`` and ``horizontalVelY.nc`` - Derived Maximum Speed at All Nodes in the Model Grid

    """

    out_filename = 'schism_max_velocity.nc'
    file_patterns = ['horizontalVelX_*.nc', 'horizontalVelY_*.nc']
    variables = ['horizontalVelX', 'horizontalVelY']
    derived_name = 'max_velocity'
    derived_time_name = 'max_velocity_times'


class MinimumSurfacePressureOutput(MinimumScalarFieldOutputCalculator):
    """
    ``out2d.nc`` - Minimum Sea-level Pressure at All Nodes in the Model Grid

    """

    out_filename = 'schism_min_pressure.nc'
    file_patterns = ['out2d_*.nc']
    variables = ['airPressure']
    derived_name = 'min_pressure'
    derived_time_name = 'min_pressure_times'


class MaximumSurfaceWindOutput(MaximumScalarFieldOutputCalculator):
    """
    ``out2d.nc`` - Maximum Surface Wind Speed at All Nodes in the Model Grid

    """

    out_filename = 'schism_max_wind.nc'
    file_patterns = ['out2d_*.nc']
    variables = ['windSpeedX', 'windSpeedY']
    derived_name = 'max_wind'
    derived_time_name = 'max_wind_times'


class FieldTimeSeriesOutput(FieldOutput, TimeSeriesOutput, ABC):
    pass


class ElevationTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``out2d.nc`` - Elevation Time Series at All Nodes in the Model Grid

    """

    out_filename = 'schism_elevation.nc'
    file_patterns = ['out2d_*.nc']
    variables = ['elevation']

    @classmethod
    def read(
        cls, filenames: Union[PathLike, List[PathLike]], names: List[str] = None
    ) -> Union[DataFrame, DataArray]:
        dataset = super().read(filenames, names)
        dataset.attrs['h0'] = dataset.minimum_depth
        return cls._set_dry_to_null(dataset)

    @classmethod
    def read_directory(
        cls, directory: PathLike, variables: List[str] = None, parallel: bool = False
    ) -> Dataset:

        dataset = super().read_directory(directory, variables, parallel)
        dataset.attrs['h0'] = dataset.minimum_depth
        return cls._set_dry_to_null(dataset)

    @classmethod
    def _set_dry_to_null(cls, dataset: Dataset) -> Dataset:
        for var in cls.variables:
            dataset[var] = dataset[var].where(dataset['dryFlagNode'] == 0, numpy.nan)

        return dataset

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

            dry_subset = dataset['dryFlagNode'] == 1

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
    ``horizontalVelX.nc`` and ``horizontalVelY.nc`` - Depth-averaged Velocity Time Series at All Nodes in the Model Grid

    """

    out_filename = 'schism_velocity.nc'
    file_patterns = ['horizontalVelX_*.nc', 'horizontalVelY_*.nc']
    variables = ['horizontalVelX', 'horizontalVelY']


class SurfacePressureTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``out2d.nc`` - Sea-level Pressure Time Series at All Nodes in the Model Grid

    """

    out_filename = 'schism_pressure.nc'
    file_patterns = ['out2d_*.nc']
    variables = ['airPressure']


class SurfaceWindTimeSeriesOutput(FieldTimeSeriesOutput):
    """
    ``out2d.nc`` - Surface Wind Velocity Time Series at All Nodes in the Model Grid

    """

    out_filename = 'schism_wind.nc'
    file_patterns = ['out2d_*.nc']
    variables = ['windSpeedX', 'windSpeedY']


class _GlobDict(UserDict):
    """A dictionary that tries to match key by unix glob if the key is not found"""

    def __getitem__(self, query_key):
        if query_key in self.data:
            return self.data[query_key]
        for stored_key in self.data:
            if fnmatch.fnmatch(query_key, stored_key):
                return self.data[stored_key]

        return super().__getitem__(query_key)


def schism_file_data_variables(cls: type = None, existing_dict=None) -> Dict[str, List[str]]:

    file_data_variables = _GlobDict()
    if existing_dict is not None:
        file_data_variables = deepcopy(existing_dict)

    if cls is None:
        cls = SchismOutput

    for subclass in cls.__subclasses__():
        try:
            for patt in subclass.file_patterns:
                file_data_variables.setdefault(patt, set()).add(subclass)
        except AttributeError:
            file_data_variables = schism_file_data_variables(subclass, file_data_variables)
    return file_data_variables


SCHISM_FILE_OUTPUTS = schism_file_data_variables()


def parse_schism_outputs(
    directory: PathLike = None, file_outputs: List[str] = None, parallel: bool = False,
) -> Dict[str, dict]:
    """
    Parse output from multiple SCHISM runs.

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
        file_outputs = SCHISM_FILE_OUTPUTS
    elif isinstance(file_outputs, Collection):
        file_outputs = {pattern: SCHISM_FILE_OUTPUTS[pattern] for pattern in file_outputs}
    elif isinstance(file_outputs, Mapping):
        file_outputs = {
            pattern: subclasses if subclasses is not None else SCHISM_FILE_OUTPUTS[pattern]
            for pattern, subclasses in file_outputs.items()
        }

    output_tree = {}
    node_info_keys = [
        'SCHISM_hgrid_node_x',
        'SCHISM_hgrid_node_y',
        'depth',
        'dryFlagNode',
        'element',
    ]
    node_info_data = xarray.Dataset()
    for basename, output_classes in file_outputs.items():
        for output_class in output_classes:
            try:
                # Some classes match multiple patterns (i.e. basename)
                if output_class in output_tree:
                    continue

                dataset = output_class.read_directory(
                    directory, variables=output_class.variables, parallel=parallel,
                )
                if len(dataset) == 0:
                    continue
                if all(
                    var in dataset.data_vars or var in dataset.coords for var in node_info_keys
                ):
                    node_info_data = dataset[node_info_keys]

                # NOTE: The dataset variable might be derived variables
                # skip_ds = False
                # for var in output_class.variables:
                #     if var in dataset.data_vars:
                #         continue
                #     skip_ds = True
                # if skip_ds:
                #     continue

                output_tree[output_class] = dataset

            except (ValueError, FileNotFoundError) as error:
                LOGGER.warning(error)

    if len(node_info_data.coords) != 0:
        for output_class, dataset in output_tree.items():
            if 'nSCHISM_hgrid_node' not in dataset.dims:
                continue
            output_tree[output_class] = dataset.merge(node_info_data)
    else:
        LOGGER.warn('No dataset found with infomration about node locations and dry nodes!')

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
        file_data_variables = SCHISM_FILE_OUTPUTS
    elif isinstance(file_data_variables, Collection):
        file_data_variables = {
            filename: SCHISM_FILE_OUTPUTS[filename] for filename in file_data_variables
        }
    elif isinstance(file_data_variables, Mapping):
        file_data_variables = {
            filename: variables if variables is not None else SCHISM_FILE_OUTPUTS[filename]
            for filename, variables in file_data_variables.items()
        }

    # parse all the inputs using built-in parser
    output_data = {'perturbations.nc': parse_vortex_perturbations(track_directory)}

    # parse all the outputs using built-in parser
    LOGGER.info(f'parsing from "{directory}"')
    parsed_files = parse_schism_outputs(
        directory=runs_directory, file_outputs=file_data_variables, parallel=parallel,
    )

    wetdry_timeseries_output = ElevationTimeSeriesOutput
    if elevation_selection is not None and wetdry_timeseries_output not in parsed_files:
        raise ValueError(f'elevation time series not found')

    output_data.update({k.out_filename: v for k, v in parsed_files.items()})

    # generate subset
    elevation_subset = None
    for output_class, file_data in parsed_files.items():
        if 'nSCHISM_hgrid_node' in file_data:
            num_nodes = len(file_data['nSCHISM_hgrid_node'])

            variable_shape_string = ', '.join(
                f'"{name}" {variable.shape}' for name, variable in file_data.items()
            )
            LOGGER.info(
                f'found {len(file_data)} variable(s) in "{output_class.file_patterns}": {variable_shape_string}'
            )

            subset = ~file_data['nSCHISM_hgrid_node'].isnull()

            if elevation_subset is not None:
                subset = numpy.logical_and(subset, elevation_subset)

            subset = numpy.logical_and(
                subset,
                output_class.subset(
                    file_data,
                    bounds=bounds,
                    maximum_depth=maximum_depth,
                    elevation_selection=elevation_selection,
                ),
            )

            if subset is not None:
                file_data = file_data.sel(nSCHISM_hgrid_node=subset)

                LOGGER.info(
                    f'subsetted {len(file_data["nSCHISM_hgrid_node"])} out of {num_nodes} total nodes ({len(file_data["nSCHISM_hgrid_node"]) / num_nodes:3.2%})'
                )

                if elevation_selection is not None:
                    elevation_subset = subset

            output_data[output_class.out_filename] = file_data

    if output_directory is not None:
        for key, file_data in output_data.items():
            basename = key
            if isinstance(key, type) and issubclass(key, SchismOutput):
                basename = key.out_filename
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
    minimum_depth: float = None,
    wind_swath: list = None,
    bounds: (float, float, float, float) = None,
    node_status_selection: dict = None,
    point_spacing: int = None,
    output_filename: PathLike = None,
):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        if node_status_selection is None:
            node_status_mask = None
        elif node_status_selection['mask'] == 'sometimes_wet':
            node_status_mask = (
                ~ds[variable].sel(run=node_status_selection['runs']).isnull().all('run'),
            )
        elif node_status_selection['mask'] == 'always_wet':
            node_status_mask = (
                ~ds[variable].sel(run=node_status_selection['runs']).isnull().any('run'),
            )
        else:
            raise ValueError(
                f'node_status_selection {node_status_selection["mask"]} unrecognized'
            )

        node_subset_mask = (
            FieldOutput.subset(
                ds['nSCHISM_hgrid_node'],
                maximum_depth=maximum_depth,
                minimum_depth=minimum_depth,
                bounds=bounds,
                wind_swath=wind_swath,
            ),
        )
        if node_status_mask is None:
            node_status_mask = node_subset_mask
        subsetted_nodes = ds['nSCHISM_hgrid_node'].values[
            numpy.logical_and(node_status_mask, node_subset_mask).squeeze()
        ]
        if point_spacing is not None:
            subsetted_nodes = subsetted_nodes[::point_spacing]
        subset = ds.sel(nSCHISM_hgrid_node=subsetted_nodes)

        # TODO: Used to be code to adjust element table if present based on node

        try:
            subset = subset.drop_sel(run='original')
        except:
            pass
        if len(subset['nSCHISM_hgrid_node']) != len(ds['nSCHISM_hgrid_node']):
            LOGGER.info(
                f'subsetted down to {len(subset["nSCHISM_hgrid_node"])} nodes ({len(subset["nSCHISM_hgrid_node"]) / len(ds["nSCHISM_hgrid_node"]):.1%})'
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
    nodes = numpy.arange(da.sizes['nSCHISM_hgrid_node'])

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


def convert_schism_output_dataset_to_adcirc_like(schism_ds: Dataset) -> Dataset:

    # TODO: Add sanity check for schism dataset

    coord_vars = []
    if 'station_index' in schism_ds.data_vars:
        # Station data
        temp_ds = schism_ds.swap_dims({'station_index': 'station'}).reset_coords(
            'station_index'
        )
        coord_vars.append('time')
    else:
        # Field data
        temp_ds = schism_ds.copy()
        if 'time' in temp_ds.data_vars:
            coord_vars.append('time')
        if 'SCHISM_hgrid_node_x' in temp_ds.data_vars:
            coord_vars.append('x')
        if 'SCHISM_hgrid_node_y' in temp_ds.data_vars:
            coord_vars.append('y')

    temp_ds = temp_ds.rename(
        **{k: v for k, v in SCHISM_ADCIRC_VAR_MAPPING.items() if k in schism_ds.data_vars}
    )
    temp_ds = temp_ds.rename(
        **{k: v for k, v in SCHISM_ADCIRC_COORD_MAPPING.items() if k in schism_ds.coords}
    )

    if 'depth' in temp_ds:
        temp_ds = temp_ds.assign_coords({'depth': temp_ds['depth']})

    if 'element' in temp_ds:
        temp_ds = temp_ds.assign_coords({'element': temp_ds['element']})

    if 'node' in temp_ds:
        temp_ds = temp_ds.assign_coords({'node': temp_ds['node']})

    if 'run' in temp_ds.coords:
        coord_vars.insert(0, 'run')

    adcirc_ds = temp_ds.set_coords(coord_vars)

    return adcirc_ds


def convert_schism_output_files_to_adcirc_like(
    directory: PathLike = None,
    file_data_variables: Dict[str, List[str]] = None,
    bounds: Tuple[float, float, float, float] = None,
    maximum_depth: float = None,
    elevation_selection: ElevationSelection = None,
    output_directory: PathLike = None,
    parallel: bool = False,
) -> None:

    if output_directory is not None:
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)

    # NOTE: Don't pass the output_directory here
    results = combine_outputs(
        directory=directory,
        file_data_variables=file_data_variables,
        bounds=bounds,
        maximum_depth=maximum_depth,
        elevation_selection=elevation_selection,
        output_directory=None,
        parallel=parallel,
    )

    output_data = {}
    for out_name, data in results.items():
        newkey = SCHISM_ADCIRC_OUT_MAPPING.get(out_name, None)
        if newkey is None:
            newkey = out_name

        if newkey != 'perturbations.nc':
            data = convert_schism_output_dataset_to_adcirc_like(data)
        output_data[newkey] = data

    if output_directory is not None:
        for out_name, file_data in output_data.items():
            output_filename = output_directory / out_name
            LOGGER.info(f'writing to "{output_filename}"')
            file_data.to_netcdf(
                output_filename,
                # encoding={
                #     variable_name: {'zlib': True} for variable_name in file_data.variables
                # },
            )

    return output_data
