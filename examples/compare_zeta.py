from pathlib import Path

from adcircpy import AdcircMesh
from adcircpy._fort15 import _Fort15
import numpy
from pandas import DataFrame

from ensemble_perturbation.parse_output import parse_adcirc_outputs
from ensemble_perturbation.perturb_adcirc import download_test_configuration

if __name__ == '__main__':
    input_directory = Path(r"C:\Users\Zachary.Burnett\Downloads\data\input")
    download_test_configuration(input_directory)
    fort14_filename = input_directory / 'fort.14'
    fort15_filename = input_directory / 'fort.15'

    output_directory = Path(r"C:\Users\Zachary.Burnett\Downloads\data\output")
    output_datasets = parse_adcirc_outputs(output_directory)

    mesh = AdcircMesh.open(fort14_filename, crs=4326)
    input_ssh = DataFrame(numpy.concatenate(
        [mesh.coords, numpy.expand_dims(mesh.values, axis=1)],
        axis=1), columns=['x', 'y', 'zeta'])

    station_types = ['NOUTE', 'NOUTV', 'NOUTM', 'NOUTC']
    ssh_stations = {}
    for station_type in station_types:
        ssh_stations.update(_Fort15.parse_stations(fort15_filename,
                                                   station_type))

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
