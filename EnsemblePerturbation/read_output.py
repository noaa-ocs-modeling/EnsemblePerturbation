import os

from netCDF4 import Dataset
import numpy
from pandas import DataFrame


def parse_adcirc_output(directory: str):
    output_files = {
        # Elevation Time Series at All Nodes in the Model Grid (fort.63)
        'fort.63.nc'  : ['zeta'],
        # Depth-averaged Velocity Time Series at All Nodes in the Model Grid (fort.64)
        'fort.64.nc'  : ['u-vel', 'v-vel'],
        # Hot Start Output (fort.67, fort.68)
        'fort.67.nc'  : ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel'],
        'fort.68.nc'  : ['zeta1', 'zeta2', 'zetad', 'u-vel', 'v-vel'],
        # Maximum Elevation at All Nodes in the Model Grid (maxele.63)
        'maxele.63.nc': ['zeta_max', 'time_of_zeta_max'],
        # Maximum Velocity at All Nodes in the Model Grid (maxvel.63)
        'maxvel.63.nc': ['vel_max', 'time_of_vel_max']
    }

    output_data = {}
    for output_file, data_variables in output_files.items():
        filename = os.path.join(directory, output_file)
        if os.path.exists(filename):
            dataset = Dataset(filename)
            coordinates = numpy.stack((dataset['x'], dataset['y'], dataset['depth']), axis=1)
            time = numpy.array(dataset['time'])

            data = {data_variable: numpy.array(dataset[data_variable])
                    for data_variable in data_variables}

            if output_file in ['fort.63.nc', 'fort.64.nc']:
                data = {'coordinates': coordinates, 'time': time, 'data': data}
            else:
                columns = ['x', 'y', 'depth'] + list(data)
                data = numpy.concatenate(
                    (
                        coordinates,
                        numpy.stack([numpy.squeeze(data_variable)
                                     for data_variable in data.values()],
                                    axis=1)
                    ),
                    axis=1)
                data = DataFrame(data, columns=columns)

            output_data[output_file] = data

    return output_data


if __name__ == '__main__':
    adcirc_output_directory = os.path.expanduser(r"~\Downloads\data\NetCDF_Shinnecock_Inlet")
    output_data = parse_adcirc_output(adcirc_output_directory)
    print('done')
