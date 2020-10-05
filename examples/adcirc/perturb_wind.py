#! /usr/bin/env python

from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry
import numpy

from ensemble_perturbation.configuration.adcirc import write_adcirc_configurations
from ensemble_perturbation.utilities import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'output'

if __name__ == '__main__':
    runs = {
        f'wind_{wind:.3}': (wind, 'wind')
        for wind in numpy.linspace(0.016, 0.08, 5)
    }

    models = {
        'atm': AtmosphericMeshEntry('atm.nc'),
        'wav': WaveMeshEntry('wav.nc'),
        'ocn': ADCIRCEntry(11)
    }

    write_adcirc_configurations(runs, INPUT_DIRECTORY, OUTPUT_DIRECTORY, **models)
    print('done')
