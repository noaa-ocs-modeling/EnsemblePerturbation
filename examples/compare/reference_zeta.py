from pathlib import Path

from adcircpy.validation import COOPS
from geopandas import GeoDataFrame
import matplotlib
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy
import pandas
from pandas import DataFrame
from pyproj import CRS, Geod
import shapely
from shapely.ops import nearest_points

from ensemble_perturbation import get_logger
from ensemble_perturbation.inputs.adcirc import download_test_configuration
from ensemble_perturbation.outputs.parse_output import fort61_stations_zeta, \
    parse_adcirc_outputs

LOGGER = get_logger('compare.zeta')

if __name__ == '__main__':
    root_directory = Path(__file__).parent.parent

    input_directory = root_directory / 'data/input'
    download_test_configuration(input_directory)
    fort14_filename = input_directory / 'fort.14'
    fort15_filename = input_directory / 'fort.15'

    output_directory = root_directory / 'data/output'
    output_datasets = parse_adcirc_outputs(output_directory)

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

    observation_color_map = matplotlib.cm.get_cmap('Blues')
    model_color_map = matplotlib.cm.get_cmap('Reds')
    error_color_map = matplotlib.cm.get_cmap('prism')
    run_index_normalizer = matplotlib.colors.Normalize(0, len(output_datasets))
    station_index_normalizer = matplotlib.colors.Normalize(0, len(
        stations_within_mesh))

    linestyles = {
        'coldstart': ':',
        'hotstart': '-'
    }

    value_figure = pyplot.figure()
    value_figure.suptitle('station zeta')
    sharing_axis = None

    value_axes = {}
    for station_index, (_, station) in \
            enumerate(stations_within_mesh.iterrows()):
        axis = value_figure.add_subplot(len(stations_within_mesh), 1,
                                        station_index + 1, sharex=sharing_axis)
        if sharing_axis is None:
            sharing_axis = axis

        value_axes[station['name']] = axis

    error_figure = pyplot.figure()
    error_figure.suptitle('zeta errors')
    error_axis = error_figure.add_subplot(1, 1, 1, sharex=sharing_axis)

    rmses = {}
    for run_index, (run_name, stages) in enumerate(output_datasets.items()):
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

            del model_times, modeled_zeta

            zeta_errors = []
            for station_index, (station_name, modeled_zeta) in \
                    enumerate(nearest_modeled_zeta.items()):
                coops_tidal_stations.add_station(station_name,
                                                 modeled_zeta['time'].min(),
                                                 modeled_zeta['time'].max())
                coops_tidal_stations.station = station_name

                coops_observed_zeta = DataFrame({
                    'time': coops_tidal_stations.datetime,
                    'zeta': coops_tidal_stations.values
                })

                observed_zeta = fort61_stations_zeta(fort61_filename,
                                                     [station_name])

                zeta_error = modeled_zeta[['time', 'zeta']] - \
                             observed_zeta[['time', 'zeta']]

                zeta_error.columns = ['time_difference', 'zeta']

                zeta_error = zeta_error[['zeta', 'time_difference']]

                zeta_error.insert(0, 'time', modeled_zeta['time'])
                zeta_error.insert(len(zeta_error.columns) - 1,
                                  'station', station_name)
                zeta_error.insert(len(zeta_error.columns),
                                  'distance', modeled_zeta['distance'])

                zeta_errors.append(zeta_error)

                observation_color = observation_color_map(
                    run_index_normalizer(run_index))
                model_color = model_color_map(run_index_normalizer(run_index))
                error_color = error_color_map(
                    station_index_normalizer(station_index))

                value_axis = value_axes[station_name]

                value_axis.plot(observed_zeta['time'], observed_zeta['zeta'],
                                color=observation_color,
                                linestyle=linestyles[stage],
                                label=f'{run_name} {stage} fort.61')
                value_axis.plot(coops_observed_zeta['time'],
                                coops_observed_zeta['zeta'],
                                color=observation_color,
                                label=f'{run_name} {stage} CO-OPS')
                value_axis.plot(observed_zeta['time'], observed_zeta['zeta'],
                                color=model_color, linestyle=linestyles[stage],
                                label=f'{run_name} {stage} model')
                error_axis.scatter(zeta_error['time'], zeta_error['zeta'],
                                   color=error_color, s=2,
                                   label=f'{run_name} {stage}')

            zeta_errors = pandas.concat(zeta_errors)

            rmses[run_name][stage] = {
                'zeta_rmse': numpy.sqrt(
                    (zeta_errors['zeta'] ** 2).mean()),
                'mean_time_difference': zeta_errors[
                    'time_difference'].mean(),
                'mean_distance': zeta_errors['distance'].mean()
            }

    value_handles = [
        Line2D([0], [0], color='b', label='Observation'),
        Line2D([0], [0], color='r', label='Model'),
        Line2D([0], [0], color='k', linestyle=linestyles['coldstart'],
               label='Coldstart'),
        Line2D([0], [0], color='k', linestyle=linestyles['hotstart'],
               label='Hotstart')
    ]

    for station_name, axis in value_axes.items():
        axis.set_title(f'station {station_name} zeta', loc='left')
        axis.hlines([0], *axis.get_xlim(), color='k', linestyle='--')
        axis.set_ylabel('zeta (m)')
        axis.legend(handles=value_handles)

    error_handles = [Line2D([0], [0],
                            color=error_color_map(
                                station_index_normalizer(station_index)),
                            label=f'station {station["name"]}')
                     for station_index, (_, station) in
                     enumerate(stations_within_mesh.iterrows())]

    error_axis.set_title(f'zeta error', loc='left')
    error_axis.hlines([0], *error_axis.get_xlim(), color='k', linestyle='--')
    error_axis.set_ylabel('zeta error (m)')
    error_axis.legend(handles=error_handles)

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

    value_figure = pyplot.figure()
    rmse_axis = value_figure.add_subplot(1, 1, 1)
    rmse_axis.set_title('zeta RMSE')
    for column in rmses:
        if 'zeta' in column:
            rmse_axis.plot(mannings_n, rmses[column], label=f'{column}')
    rmse_axis.set_xlabel('Manning\'s N')
    rmse_axis.set_ylabel('zeta RMSE (m)')
    rmse_axis.legend()

    pyplot.show()

    print('done')
