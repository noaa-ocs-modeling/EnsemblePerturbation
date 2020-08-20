from pathlib import Path

from adcircpy import AdcircMesh
import numpy
import pandas
from pandas import DataFrame
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points

from ensemble_perturbation import get_logger
from ensemble_perturbation.inputs.perturb_adcirc import \
    download_test_configuration
from ensemble_perturbation.outputs.parse_output import parse_adcirc_outputs

LOGGER = get_logger('compare.zeta')

if __name__ == '__main__':
    input_directory = Path(__file__) / '../data/input'
    download_test_configuration(input_directory)
    fort14_filename = input_directory / 'fort.14'
    fort15_filename = input_directory / 'fort.15'

    output_directory = Path(__file__) / '../data/output'
    output_datasets = parse_adcirc_outputs(output_directory)

    mesh = AdcircMesh.open(fort14_filename, crs=4326)

    rmses = {}
    for run, stages in output_datasets.items():
        if run not in rmses:
            rmses[run] = {}
        for stage, datasets in stages.items():
            ssh_stations = datasets['fort.61.nc']
            modeled_ssh = datasets['fort.63.nc']['data']['zeta']
            max_modeled_ssh = datasets['maxele.63.nc']

            station_coordinates = ssh_stations['coordinates']
            stations_ssh = numpy.array(ssh_stations['data']['zeta']).T
            station_ids = [int(station_name.tobytes().decode())
                           for station_name in
                           numpy.array(ssh_stations['data']['station_name'])]
            ssh_stations = {
                station_ids[index]: {
                    'point': Point(*station_coordinates[index]),
                    'ssh': DataFrame(
                        {'time': ssh_stations['time'],
                         'zeta': stations_ssh[index]})
                }
                for index in range(len(station_coordinates))
            }

            mesh_hull = MultiPoint(
                numpy.array(max_modeled_ssh[['x', 'y']])).convex_hull
            stations_within_mesh = {
                station_id: station
                for station_id, station in ssh_stations.items()
                if station['point'].within(mesh_hull)
            }

            nearest_modeled_ssh = {}
            for station_id, station in stations_within_mesh.items():
                nearest_point = nearest_points(station['point'], MultiPoint(
                    numpy.array(max_modeled_ssh['geometry'])))[1]
                nearest_modeled_ssh[station_id] = max_modeled_ssh.cx[
                    nearest_point.x, nearest_point.y]

            errors = {}
            for station_id, station in stations_within_mesh.items():
                modeled_ssh = nearest_modeled_ssh[station_id]
                modeled_ssh.columns = ['x', 'y', 'depth', 'zeta', 'time',
                                       'geometry']
                observed_ssh = station['ssh'].iloc[station['ssh']['time'].sub(
                    modeled_ssh['time'].item()).abs().idxmin()]
                difference = (modeled_ssh[['time', 'zeta']] -
                              observed_ssh).abs()
                difference.index = [station_id]
                errors[station_id] = difference
            errors = pandas.concat(errors.values())

            rmses[run][stage] = {
                'zeta': numpy.sqrt(numpy.nanmean(errors['zeta'].item() ** 2)),
                'time_difference': errors['time'].item()
            }

    rmses = DataFrame({'run': list(rmses.keys()),
                       **{f'{stage}_{difference}': [rmse[stage][difference]
                                                    for rmse in rmses.values()]
                          for stage in ['coldstart', 'hotstart']
                          for difference in ['zeta', 'time_difference']
                          }
                       })

    rmses.to_csv(output_directory / 'rmse.csv', index=False)

    print('done')
