#! /usr/bin/env python
from coupledmodeldriver.generate import generate_adcirc_configuration
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, \
    WaveWatch3MeshEntry
import numpy

from ensembleperturbation.utilities import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'output'

if __name__ == '__main__':
    runs = {f'wind_{wind:.3}': (wind, 'wind') for wind in numpy.linspace(0.016, 0.08, 5)}

    models = {
        'atm': AtmosphericMeshEntry('atm.nc'),
        'wav': WaveWatch3MeshEntry('wav.nc'),
        'ocn': ADCIRCEntry(11),
    }

    generate_adcirc_configuration(runs, INPUT_DIRECTORY, OUTPUT_DIRECTORY, **models)
    print('done')
