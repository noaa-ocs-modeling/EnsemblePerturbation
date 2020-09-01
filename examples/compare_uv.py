from pathlib import Path

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
from ensemble_perturbation.outputs.parse_output import fort62_stations_uv, \
    parse_adcirc_outputs

LOGGER = get_logger('compare.uv')


def insert_magnitude_components(dataframe: DataFrame,
                                u: str = 'u',
                                v: str = 'v',
                                magnitude: str = 'magnitude',
                                direction: str = 'direction',
                                velocity_index: int = None,
                                direction_index: int = None):
    if velocity_index is None:
        velocity_index = len(dataframe.columns)
    if direction_index is None:
        direction_index = velocity_index + 1
    dataframe.insert(velocity_index, magnitude, numpy.hypot(dataframe[u],
                                                            dataframe[v]))
    dataframe.insert(direction_index, direction, numpy.arctan2(dataframe[u],
                                                               dataframe[v]))


if __name__ == '__main__':
    input_directory = Path(__file__) / '../data/input'
    download_test_configuration(input_directory)
    fort14_filename = input_directory / 'fort.14'
    fort15_filename = input_directory / 'fort.15'

    output_directory = Path(__file__) / '../data/output'
    output_datasets = parse_adcirc_outputs(output_directory)

    crs = CRS.from_epsg(4326)
    ellipsoid = crs.datum.to_json_dict()['ellipsoid']
    geodetic = Geod(a=ellipsoid['semi_major_axis'],
                    rf=ellipsoid['inverse_flattening'])

    first_coldstart = output_datasets[list(output_datasets)[0]]['coldstart']

    stations = first_coldstart['fort.62.nc']
    mesh = first_coldstart['maxvel.63.nc']

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
    error_color_map = matplotlib.cm.get_cmap('Spectral')
    color_normalizer = matplotlib.colors.Normalize(0, len(output_datasets))

    linestyles = {
        'coldstart': ':',
        'hotstart': '-'
    }

    figure = pyplot.figure()
    value_axis = figure.add_subplot(2, 1, 1)
    error_axis = figure.add_subplot(2, 1, 2, sharex=value_axis)

    rmses = {}
    for run_index, (run_name, stages) in enumerate(output_datasets.items()):
        if run_name not in rmses:
            rmses[run_name] = {}

        for stage, datasets in stages.items():
            fort62_filename = output_directory / run_name / stage / 'fort.62.nc'

            model_times = datasets['fort.64.nc']['time']
            modeled_u = datasets['fort.64.nc']['data']['u-vel']
            modeled_v = datasets['fort.64.nc']['data']['v-vel']

            nearest_modeled_uv = {
                nearest_mesh_vertex['station']: GeoDataFrame({
                    'time': model_times,
                    'u': modeled_u[:, nearest_mesh_point_index],
                    'v': modeled_v[:, nearest_mesh_point_index],
                    'distance': nearest_mesh_vertex['distance']
                }, geometry=[nearest_mesh_vertex.geometry
                             for _ in model_times])
                for nearest_mesh_point_index, nearest_mesh_vertex in
                nearest_mesh_vertices.iterrows()
            }

            del model_times, modeled_u, modeled_v

            uv_errors = []
            for station_name, modeled_uv in nearest_modeled_uv.items():
                observed_uv = fort62_stations_uv(fort62_filename,
                                                 [station_name])

                uv_error = modeled_uv[['time', 'u', 'v']] - \
                           observed_uv[['time', 'u', 'v']]

                uv_error.columns = ['time_difference', 'u', 'v']

                uv_error = uv_error[['u', 'v',
                                     'time_difference']]

                uv_error.insert(0, 'time', modeled_uv['time'])
                uv_error.insert(len(uv_error.columns) - 1, 'station',
                                station_name)
                uv_error.insert(len(uv_error.columns), 'distance',
                                modeled_uv['distance'])

                insert_magnitude_components(modeled_uv)
                insert_magnitude_components(observed_uv)
                insert_magnitude_components(uv_error, velocity_index=3)

                uv_errors.append(uv_error)

                observation_color = observation_color_map(
                    color_normalizer(run_index))
                model_color = model_color_map(color_normalizer(run_index))
                error_color = error_color_map(color_normalizer(run_index))

                value_axis.plot(observed_uv['time'], observed_uv['magnitude'],
                                color=observation_color,
                                linestyle=linestyles[stage],
                                label=f'{run_name} {stage} fort.62 magnitude')
                value_axis.plot(modeled_uv['time'], modeled_uv['magnitude'],
                                color=model_color, linestyle=linestyles[stage],
                                label=f'{run_name} {stage} model magnitude')
                error_axis.scatter(uv_error['time'], uv_error['magnitude'],
                                   color=error_color, s=2,
                                   label=f'{run_name} {stage} magnitude error')

            uv_errors = pandas.concat(uv_errors)

            rmses[run_name][stage] = {
                'uv_rmse': numpy.sqrt((uv_errors['magnitude'] **
                                       2).mean()),
                'u_rmse': numpy.sqrt((uv_errors['u'] ** 2).mean()),
                'v_rmse': numpy.sqrt((uv_errors['v'] ** 2).mean()),
                'mean_time_difference': uv_errors[
                    'time_difference'].mean(),
                'mean_distance': uv_errors['distance'].mean()
            }

    value_handles = [
        Line2D([0], [0], color='b', label='Observation'),
        Line2D([0], [0], color='g', label='Model'),
        Line2D([0], [0], color='k', linestyle=linestyles['coldstart'],
               label='Coldstart'),
        Line2D([0], [0], color='k', linestyle=linestyles['hotstart'],
               label='Hotstart')
    ]

    value_axis.set_title(f'uv magnitude', loc='left')
    value_axis.hlines([0], *value_axis.get_xlim(), color='k', linestyle='--')
    value_axis.set_ylabel('uv (m/s)')
    value_axis.legend(handles=value_handles)

    error_handles = [Line2D([0], [0],
                            color=error_color_map(color_normalizer(index)),
                            label=run_name)
                     for index, run_name in enumerate(output_datasets)]

    error_axis.set_title(f'uv magnitude error', loc='left')
    error_axis.hlines([0], *error_axis.get_xlim(), color='k', linestyle='--')
    error_axis.set_ylabel('uv error (m/s)')
    error_axis.legend(handles=error_handles)

    rmses = DataFrame({
        'run': list(rmses.keys()),
        **{f'{stage}_{value}': [rmse[stage][value]
                                for rmse in rmses.values()]
           for stage in ['coldstart', 'hotstart']
           for value in ['uv_rmse', 'mean_time_difference', 'mean_distance']
           }
    })

    rmses.to_csv(output_directory / 'rmse.csv', index=False)

    mannings_n = [float(run.replace('mannings_n_', ''))
                  for run in rmses['run']]

    figure = pyplot.figure()
    figure.suptitle('RMSE')
    rmse_axis = figure.add_subplot(1, 1, 1)
    for column in rmses:
        if 'uv' in column:
            rmse_axis.plot(mannings_n, rmses[column], label=f'{column}')
            rmse_axis.set_xlabel('Manning\'s N')
            rmse_axis.set_ylabel('uv RMSE (m)')
            rmse_axis.legend()

    pyplot.show()

    print('done')
