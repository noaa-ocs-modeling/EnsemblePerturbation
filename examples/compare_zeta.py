from pathlib import Path

from adcircpy.validation import COOPS
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

    stations_within_mesh = stations[stations.within(
        mesh.unary_union.convex_hull.buffer(0.01))]

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
            'station': station['name'],
            'station_x': station.geometry.x,
            'station_y': station.geometry.y,
            'distance': distance
        }, geometry=[nearest_mesh_point], index=[nearest_mesh_point_index]))
    nearest_mesh_vertices = pandas.concat(nearest_mesh_vertices)

    axes = {}
    for station_index, (_, station) in enumerate(
            stations_within_mesh.iterrows()):
        zeta_axis = figure.add_subplot(
            len(stations_within_mesh) * 2, 1, station_index * 2 + 1,
            sharex=sharing_axis)
        if sharing_axis is None:
            sharing_axis = zeta_axis
        difference_axis = figure.add_subplot(
            len(stations_within_mesh) * 2, 1, station_index * 2 + 2,
            sharex=sharing_axis)

        axes[station['name']] = {
            'zeta': zeta_axis,
            'zeta_difference': difference_axis
        }

    rmses = {}
    for run_name, stages in output_datasets.items():
        if run_name not in rmses:
            rmses[run_name] = {}

        for stage, datasets in stages.items():
            fort61_filename = output_directory / run_name / stage / 'fort.61.nc'

            coops_tidal_stations = COOPS.TidalStations()

            model_times = datasets['fort.63.nc']['time']
            modeled_zeta = datasets['fort.63.nc']['data']['zeta']

            nearest_modeled_zeta = {
                nearest_mesh_vertex['station']: GeoDataFrame({
                    'time': model_times,
                    'zeta': modeled_zeta[:, nearest_mesh_point_index],
                    'distance': nearest_mesh_vertex['distance']
                }, geometry=[nearest_mesh_vertex.geometry
                             for _ in model_times])
                for nearest_mesh_point_index, nearest_mesh_vertex in
                nearest_mesh_vertices.iterrows()
            }

            del model_times
            del modeled_zeta

            zeta_differences = []
            for station_index, (station_name, modeled_zeta) in enumerate(
                    nearest_modeled_zeta.items()):
                coops_tidal_stations.add_station(station_name,
                                                 modeled_zeta['time'].min(),
                                                 modeled_zeta['time'].max())
                coops_tidal_stations.station = station_name

                coops_observed_zeta = DataFrame({
                    'time': coops_tidal_stations.datetime,
                    'zeta': coops_tidal_stations.values
                })

                observed_zeta = parse_fort61_stations(fort61_filename,
                                                      [station_name])
                observed_zeta = parse_fort61_stations(fort61_filename,
                                                      [station_name])

                zeta_difference = (modeled_zeta[['time', 'zeta']] -
                                   observed_zeta[['time', 'zeta']]).abs()

                zeta_difference.columns = ['time_difference', 'zeta']

                zeta_difference = zeta_difference[
                    reversed(zeta_difference.columns)]

                zeta_difference.insert(0, 'time', modeled_zeta['time'])
                zeta_difference.insert(len(zeta_difference.columns) - 1,
                                       'station', station_name)
                zeta_difference.insert(len(zeta_difference.columns),
                                       'distance', modeled_zeta['distance'])

                zeta_differences.append(zeta_difference)

                zeta_axis = axes[station_name]['zeta']
                difference_axis = axes[station_name]['zeta_difference']

                zeta_axis.plot(observed_zeta['time'],
                               observed_zeta['zeta'],
                               label=f'{run_name} {stage} fort.61')
                zeta_axis.plot(coops_observed_zeta['time'],
                               coops_observed_zeta['zeta'],
                               label=f'{run_name} {stage} CO-OPS')
                zeta_axis.plot(modeled_zeta['time'],
                               modeled_zeta['zeta'],
                               label=f'{run_name} {stage} model')
                difference_axis.plot(zeta_difference['time'],
                                     zeta_difference['zeta'],
                                     label=f'{run_name} {stage}')

            zeta_differences = pandas.concat(zeta_differences)

            rmses[run_name][stage] = {
                'zeta_rmse': numpy.sqrt(
                    (zeta_differences['zeta'] ** 2).mean()),
                'mean_time_difference': zeta_differences[
                    'time_difference'].mean(),
                'mean_distance': zeta_differences['distance'].mean()
            }

    for station_name, station_axes in axes.items():
        station_axes['zeta'].set_title(f'station {station_name} zeta',
                                       loc='left')
        station_axes['zeta_difference'].set_title(
            f'station {station_name} zeta difference', loc='left')
        station_axes['zeta'].set_ylabel('zeta (m)')
        station_axes['zeta_difference'].set_ylabel('zeta difference (m)')
        station_axes['zeta'].legend(bbox_to_anchor=(1, 1), loc='upper left',
                                    fontsize='xx-small')
        station_axes['zeta_difference'].legend(bbox_to_anchor=(1, 1),
                                               loc='upper left',
                                               fontsize='xx-small')

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
    rmse_axis = figure.add_subplot(1, 1, 1)
    rmse_axis.suptitle('RMSE')
    for column in rmses:
        if 'zeta' in column:
            rmse_axis.plot(mannings_n, rmses[column], label=f'{column}')
    rmse_axis.set_xlabel('Manning\'s N')
    rmse_axis.set_ylabel('zeta RMSE (m)')
    rmse_axis.legend()

    pyplot.show()

    print('done')
