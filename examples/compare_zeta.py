from pathlib import Path

from geopandas import GeoDataFrame
from matplotlib import pyplot
import numpy
import pandas
from pandas import DataFrame
from pyproj import CRS, Geod
import shapely
from shapely.ops import nearest_points

from ensemble_perturbation import get_logger
from ensemble_perturbation.inputs.adcirc import download_test_configuration
from ensemble_perturbation.outputs.parse_output import parse_adcirc_outputs, \
    parse_fort61_stations

LOGGER = get_logger('compare.zeta')

if __name__ == '__main__':
    input_directory = Path(__file__) / '../data/input'
    download_test_configuration(input_directory)
    fort14_filename = input_directory / 'fort.14'
    fort15_filename = input_directory / 'fort.15'

    output_directory = Path(__file__) / '../data/output'
    output_datasets = parse_adcirc_outputs(output_directory)

    figure = pyplot.figure()
    sharing_axis = None

    crs = CRS.from_epsg(4326)
    ellipsoid = crs.datum.to_json_dict()['ellipsoid']
    geodetic = Geod(a=ellipsoid['semi_major_axis'],
                    rf=ellipsoid['inverse_flattening'])

    first_coldstart = output_datasets[list(output_datasets)[0]]['coldstart']

    stations = first_coldstart['fort.61.nc']
    mesh = first_coldstart['maxele.63.nc']

    stations_within_mesh = stations[
        stations.within(mesh.unary_union.convex_hull)]

    nearest_mesh_vertices = []
    for station_index, station in stations_within_mesh.iterrows():
        nearest_mesh_point = shapely.ops.nearest_points(
            station.geometry,
            mesh.unary_union
        )[1]

        distance = geodetic.line_length(
            [station.geometry.x, nearest_mesh_point.x],
            [station.geometry.y, nearest_mesh_point.y])
        nearest_mesh_point_index = mesh.cx[nearest_mesh_point.x,
                                           nearest_mesh_point.y].index.item()
        nearest_mesh_vertices.append(GeoDataFrame({
            'name': station['name'],
            'station_x': station.geometry.x,
            'station_y': station.geometry.y,
            'distance': distance
        }, geometry=[nearest_mesh_point], index=[nearest_mesh_point_index]))
    nearest_mesh_vertices = pandas.concat(nearest_mesh_vertices)

    axes = {}
    for station_index, (_, station) in enumerate(
            stations_within_mesh.iterrows()):
        elevation_axis = figure.add_subplot(
            len(stations_within_mesh) * 2, 1, station_index + 1,
            sharex=sharing_axis)

        if sharing_axis is None:
            sharing_axis = elevation_axis

        difference_axis = figure.add_subplot(
            len(stations_within_mesh) * 2, 1, station_index + 2,
            sharex=sharing_axis)

        axes[station['name']] = {
            'elevation': elevation_axis,
            'difference': difference_axis
        }

    rmses = {}
    for run_name, stages in output_datasets.items():
        if run_name not in rmses:
            rmses[run_name] = {}

        for stage, datasets in stages.items():
            fort61_filename = output_directory / run_name / stage / 'fort.61.nc'

            model_times = datasets['fort.63.nc']['time']
            modeled_ssh = datasets['fort.63.nc']['data']['zeta']

            nearest_modeled_ssh = {
                nearest_mesh_vertex['name']: GeoDataFrame({
                    'time': model_times,
                    'zeta': modeled_ssh[:, nearest_mesh_point_index],
                    'distance': nearest_mesh_vertex['distance']
                }, geometry=[nearest_mesh_vertex.geometry
                             for _ in model_times])
                for nearest_mesh_point_index, nearest_mesh_vertex in
                nearest_mesh_vertices.iterrows()
            }

            del model_times
            del modeled_ssh

            differences = []
            for station_index, (station_name, modeled_ssh) in enumerate(
                    nearest_modeled_ssh.items()):
                observed_ssh = parse_fort61_stations(fort61_filename,
                                                     [station_name])

                difference = (modeled_ssh[['time', 'zeta']] -
                              observed_ssh[['time', 'zeta']]).abs()

                difference.columns = ['time_difference', 'zeta']

                difference = difference[reversed(difference.columns)]

                difference.insert(0, 'time', modeled_ssh['time'])
                difference.insert(len(difference.columns) - 1, 'station',
                                  station_name)
                difference.insert(len(difference.columns), 'distance',
                                  modeled_ssh['distance'])

                differences.append(difference)

                axes[station_name]['elevation'].plot(observed_ssh['time'],
                                                     observed_ssh['zeta'],
                                                     label=f'{run_name} {stage} observation')
                axes[station_name]['elevation'].plot(modeled_ssh['time'],
                                                     modeled_ssh['zeta'],
                                                     label=f'{run_name} {stage} model')
                axes[station_name]['difference'].plot(difference['time'],
                                                      difference['zeta'],
                                                      label=f'{run_name} {stage}')

            differences = pandas.concat(differences)

            rmses[run_name][stage] = {
                'zeta_rmse': numpy.sqrt((differences['zeta'] ** 2).mean()),
                'mean_time_difference': differences['time_difference'].mean(),
                'mean_distance': differences['distance'].mean()
            }

    for station_name, station_axes in axes.items():
        station_axes['elevation'].set_title(
            f'station {station_name} elevation')
        station_axes['difference'].set_title(
            f'station {station_name} elevation difference')
        station_axes['elevation'].set_ylabel('zeta (m)')
        station_axes['difference'].set_ylabel('zeta difference (m)')
        station_axes['elevation'].legend(bbox_to_anchor=(1, 1),
                                         loc='upper left', fontsize='xx-small')
        station_axes['difference'].legend(bbox_to_anchor=(1, 1),
                                          loc='upper left',
                                          fontsize='xx-small')

    # pyplot.show()

    rmses = DataFrame({
        'run': list(rmses.keys()),
        **{f'{stage}_{value}': [rmse[stage][value]
                                for rmse in rmses.values()]
           for stage in ['coldstart', 'hotstart']
           for value in ['zeta_rmse', 'mean_time_difference', 'mean_distance']
           }
    })

    rmses.to_csv(output_directory / 'rmse.csv', index=False)

    mannings_n = [float(run.replace('mannings_n_', ''))
                  for run in rmses['run']]

    figure = pyplot.figure()
    elevation_axis = figure.add_subplot(1, 1, 1)
    elevation_axis.suptitle('RMSE')
    for column in rmses:
        if 'zeta' in column:
            elevation_axis.plot(mannings_n, rmses[column], label=f'{column}')
    elevation_axis.set_xlabel('Manning\'s N')
    elevation_axis.set_ylabel('zeta RMSE (m)')
    elevation_axis.legend()

    pyplot.show()

    print('done')
