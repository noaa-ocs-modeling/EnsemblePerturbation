from pathlib import Path
from typing import Union

from netCDF4 import Dataset, Variable
import numpy
from pandas import DataFrame

ADCIRC_OUTPUT_DATA_VARIABLES = {
    # Elevation Time Series at All Nodes in the Model Grid (fort.63)
    'fort.63.nc': ['zeta'],
    # Depth-averaged Velocity Time Series at All Nodes in the Model Grid (fort.64)
    'fort.64.nc': ['u-vel', 'v-vel'],
    # Hot Start Output (fort.67, fort.68)
    'fort.67.nc': ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel'],
    'fort.68.nc': ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel'],
    # Maximum Elevation at All Nodes in the Model Grid (maxele.63)
    'maxele.63.nc': ['zeta_max', 'time_of_zeta_max'],
    # Maximum Velocity at All Nodes in the Model Grid (maxvel.63)
    'maxvel.63.nc': ['vel_max', 'time_of_vel_max']
}


def decode_time(variable: Variable) -> numpy.array:
    unit, direction, base_date = variable.units.split(' ', 2)
    intervals = {
        'years': 'Y',
        'months': 'M',
        'days': 'D',
        'hours': 'h',
        'minutes': 'm',
        'seconds': 's'
    }
    return numpy.datetime64(base_date) + numpy.array(variable).astype(
        f'timedelta64[{intervals[unit]}]')


def parse_adcirc_netcdf(
        filename: str,
        data_variables: [str] = None
) -> Union[dict, DataFrame]:
    """
    Parse ADCIRC output files

    :param filename: file path to ADCIRC NetCDF output
    :param data_variables: list of data variables to extract
    :return: parsed data
    """

    if not isinstance(filename, Path):
        filename = Path(filename)
    basename = filename.parts[-1]

    if data_variables is None:
        data_variables = ADCIRC_OUTPUT_DATA_VARIABLES[basename]

    dataset = Dataset(filename)

    coordinates = numpy.stack((dataset['x'], dataset['y'], dataset['depth']),
                              axis=1)

    times = decode_time(dataset['time'])

    data = {data_variable: dataset[data_variable]
            for data_variable in data_variables}

    if basename in ['fort.63.nc', 'fort.64.nc']:
        data = {'coordinates': coordinates, 'time': times, 'data': data}
    else:
        columns = ['x', 'y', 'depth'] + list(data)
        for array in data.values():
            assert numpy.prod(array.shape) > 0, \
                f'invalid data shape "{array.shape}"'
        data = numpy.concatenate(
            (
                coordinates,
                numpy.stack([numpy.squeeze(data_variable)
                             for data_variable in data.values()],
                            axis=1)
            ),
            axis=1)
        data = DataFrame(data, columns=columns)

    return data


def parse_adcirc_output(
        directory: str,
        file_data_variables: [str] = None
) -> {str: Union[dict, DataFrame]}:
    """
    Parse ADCIRC output files

    :param directory: path to directory containing ADCIRC output files in NetCDF format
    :param file_data_variables: output files to parse
    :return: dictionary of output data
    """

    if not isinstance(directory, Path):
        directory = Path(directory)

    if file_data_variables is None:
        file_data_variables = ADCIRC_OUTPUT_DATA_VARIABLES
    else:
        file_data_variables = {filename: ADCIRC_OUTPUT_DATA_VARIABLES[filename]
                               for filename in file_data_variables}

    output_data = {}
    for output_filename in directory.glob('*.nc'):
        basename = output_filename.parts[-1]
        output_data[basename] = parse_adcirc_netcdf(output_filename,
                                                    file_data_variables[
                                                        basename])

    return output_data


def parse_adcirc_outputs(directory: str) -> {str: dict}:
    """
    Parse output from multiple ADCIRC runs.

    :param directory: directory containing run output directories
    :return: dictionary of file tree containing parsed data
    """

    if not isinstance(directory, Path):
        directory = Path(directory)

    output_datasets = {}
    for filename in directory.glob('**/*.nc'):
        parts = Path(str(filename).split(str(directory))[-1]).parts[1:]
        tree = output_datasets
        for part_index in range(len(parts)):
            part = parts[part_index]
            if part_index < len(parts) - 1:
                if part not in tree:
                    tree[part] = {}
                tree = tree[part]
            else:
                tree[part] = parse_adcirc_netcdf(filename)

    return output_datasets
