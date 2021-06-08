#! /usr/bin/env python

from coupledmodeldriver.generate.adcirc import generate_adcirc_configuration
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry
import numpy

from ensembleperturbation.utilities import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'output'

if __name__ == '__main__':
    runs = {f'tides_{tides:.3}': (tides, 'tides') for tides in numpy.linspace(0.016, 0.08, 5)}

    models = {
        'atm': AtmosphericMeshEntry('../data/wind_atm_fin_ch_time_vec.nc'),
        'wav': WaveMeshEntry('../data/ww3.Constant.20151214_sxy_ike_date.nc'),
        'ocn': ADCIRCEntry(11),
    }

    generate_adcirc_configuration(runs, INPUT_DIRECTORY, OUTPUT_DIRECTORY, **models)
    print('done')
