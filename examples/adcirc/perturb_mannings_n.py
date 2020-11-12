#! /usr/bin/env python
from datetime import datetime, timedelta

from nemspy import ModelingSystem
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry
import numpy

from ensemble_perturbation.configuration.adcirc import write_adcirc_configurations
from ensemble_perturbation.utilities import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'configuration' / 'perturbation'

if __name__ == '__main__':
    range = [0.016, 0.08]
    mean = numpy.mean(range)
    std = mean / 3

    runs = {
        f'mannings_n_{mannings_n:.3}': (mannings_n, 'mannings_n_at_sea_floor')
        for mannings_n in numpy.random.normal(mean, std, 5)
    }

    nems = ModelingSystem(
        start_time=datetime(2020, 6, 1),
        duration=timedelta(days=7),
        interval=timedelta(hours=1),
        atm=AtmosphericMeshEntry('../../data/wind_atm_fin_ch_time_vec.nc'),
        wav=WaveMeshEntry('../../data/ww3.Constant.20151214_sxy_ike_date.nc'),
        ocn=ADCIRCEntry(11),
    )

    nems.connect('ATM', 'OCN')
    nems.connect('WAV', 'OCN')
    nems.sequence = [
        'ATM -> OCN',
        'WAV -> OCN',
        'ATM',
        'WAV',
        'OCN',
    ]

    write_adcirc_configurations(
        nems,
        runs,
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
        name='mannings_n_perturbation',
        email_address='zachary.burnett@noaa.gov',
        tacc=True,
        spinup=timedelta(hours=6),
    )
    print('done')
