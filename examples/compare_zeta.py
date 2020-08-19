from datetime import datetime
from pathlib import Path

from adcircpy import AdcircMesh
import numpy
from pandas import DataFrame
import requests

from ensemble_perturbation import get_logger
from ensemble_perturbation.inputs.perturb_adcirc import \
    download_test_configuration
from ensemble_perturbation.outputs.parse_output import parse_adcirc_outputs

LOGGER = get_logger('comparison')

if __name__ == '__main__':
    input_directory = Path(__file__) / '../data/input'
    download_test_configuration(input_directory)
    fort14_filename = input_directory / 'fort.14'
    fort15_filename = input_directory / 'fort.15'

    output_directory = Path(__file__) / '../data/output'
    output_datasets = parse_adcirc_outputs(output_directory)

    mesh = AdcircMesh.open(fort14_filename, crs=4326)
    input_ssh = DataFrame(numpy.concatenate(
        [mesh.coords, numpy.expand_dims(mesh.values, axis=1)],
        axis=1), columns=['x', 'y', 'zeta'])

    station_ids = [8512769, 8510560, 8513825, 8510448, 8510321]
    start_date = datetime(2020, 8, 18)
    end_date = datetime.today()

    stations = {}
    for station_id in station_ids:
        station_url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={start_date:%Y%m%d}&end_date={end_date:%Y%m%d}&station={station_id}&product=water_level&datum=MLLW&time_zone=gmt&units=english&format=json"
        station = requests.get(station_url).json()
        if 'error' in station:
            LOGGER.info(station['error'])
            continue
        station['data'] = DataFrame.from_dict({
            index: station_data for index, station_data in
            enumerate(station['data'])
        }, orient='index')
        stations[station_id] = station

    rmses = {}
    for run, stages in output_datasets.items():
        if run not in rmses:
            rmses[run] = {}
        for stage, datasets in stages.items():
            modeled_ssh = datasets['fort.63.nc']['data']['zeta']
            max_modeled_ssh = datasets['maxele.63.nc'][['x', 'y', 'zeta_max']]

            rmses[run][stage] = numpy.sqrt(numpy.mean(
                (max_modeled_ssh['zeta_max'].to_numpy() -
                 input_ssh['zeta'].to_numpy()) ** 2))

    print('done')
