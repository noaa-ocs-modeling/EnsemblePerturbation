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
    error_color_map = matplotlib.cm.get_cmap('prism')
    run_index_normalizer = matplotlib.colors.Normalize(0, len(output_datasets))
    station_index_normalizer = matplotlib.colors.Normalize(0, len(
        stations_within_mesh))

    components = 'u', 'v'

    linestyles = {
        'coldstart': ':',
        'hotstart': '-'
    }

    value_figure = pyplot.figure()
    value_figure.suptitle('station uv')
    sharing_axis = None

    value_axes = {}
    for station_index, (_, station) in \
            enumerate(stations_within_mesh.iterrows()):
        u_axis = value_figure.add_subplot(len(stations_within_mesh) * 2, 1,
                                          station_index * 2 + 1,
                                          sharex=sharing_axis)
        if sharing_axis is None:
            sharing_axis = u_axis
        v_axis = value_figure.add_subplot(len(stations_within_mesh) * 2, 1,
                                          station_index * 2 + 2,
                                          sharex=sharing_axis)

        value_axes[station['name']] = dict(zip(components, (u_axis, v_axis)))

    error_figure = pyplot.figure()
    error_figure.suptitle('uv errors')
    error_axes = {component: error_figure.add_subplot(len(components), 1,
                                                      index + 1,
                                                      sharex=sharing_axis)
                  for index, component in enumerate(components)}

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
            for station_index, (station_name, modeled_uv) in \
                    enumerate(nearest_modeled_uv.items()):
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
                    run_index_normalizer(run_index))
                model_color = model_color_map(run_index_normalizer(run_index))
                error_color = error_color_map(
                    station_index_normalizer(station_index))

                axes = value_axes[station_name]
                for component, axis in axes.items():
                    axis.plot(observed_uv['time'], observed_uv[component],
                              color=observation_color,
                              linestyle=linestyles[stage],
                              label=f'{run_name} {stage} fort.62 {component}')
                    axis.plot(modeled_uv['time'], modeled_uv[component],
                              color=model_color, linestyle=linestyles[stage],
                              label=f'{run_name} {stage} model {component}')
                for component, axis in error_axes.items():
                    axis.scatter(uv_error['time'], uv_error[component],
                                 color=error_color, s=2,
                                 label=f'{run_name} {stage} {component} error')

            uv_errors = pandas.concat(uv_errors)

            rmses[run_name][stage] = {
                'uv_rmse': numpy.sqrt((uv_errors['magnitude'] **
                                       2).mean()),
                'u_rmse': numpy.sqrt((uv_errors['u'] ** 2).mean()),
                'v_rmse': numpy.sqrt((uv_errors['v'] ** 2).mean()),
                'mean_time_difference': uv_errors['time_difference'].mean(),
                'mean_distance': uv_errors['distance'].mean()
            }

    value_handles = [
        Line2D([0], [0], color='b', label='Observation'),
        Line2D([0], [0], color='r', label='Model'),
        Line2D([0], [0], color='k', linestyle=linestyles['coldstart'],
               label='Coldstart'),
        Line2D([0], [0], color='k', linestyle=linestyles['hotstart'],
               label='Hotstart')
    ]

    for station_name, axes in value_axes.items():
        for component, axis in axes.items():
            axis.set_title(f'station {station_name} {component}', loc='left')
            axis.hlines([0], *axis.get_xlim(), color='k', linestyle='--')
            axis.set_ylabel(f'{component} (m/s)')
            axis.legend(handles=value_handles)

    error_handles = [Line2D([0], [0],
                            color=error_color_map(
                                station_index_normalizer(station_index)),
                            label=f'station {station["name"]}')
                     for station_index, (_, station) in
                     enumerate(stations_within_mesh.iterrows())]

    for component, axis in error_axes.items():
        axis.set_title(f'{component} error', loc='left')
        axis.hlines([0], *axis.get_xlim(), color='k', linestyle='--')
        axis.set_ylabel(f'{component} error (m/s)')
        axis.legend(handles=error_handles)

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
    figure.suptitle('uv magnitude RMSE')
    rmse_axis = figure.add_subplot(1, 1, 1)
    for column in rmses:
        if 'uv' in column:
            rmse_axis.plot(mannings_n, rmses[column], label=f'{column}')
            rmse_axis.set_xlabel('Manning\'s N')
            rmse_axis.set_ylabel('uv RMSE (m)')
            rmse_axis.legend()

    pyplot.show()

    print('done')
