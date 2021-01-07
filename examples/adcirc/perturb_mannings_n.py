#! /usr/bin/env python

from datetime import datetime, timedelta

from nemspy import ModelingSystem
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry
import numpy

from ...ensemble_perturbation.configuration.adcirc import download_shinnecock_mesh, write_adcirc_configurations
from ...ensemble_perturbation.configuration.job_script import HPC
from ...ensemble_perturbation.utilities import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input' / 'hsofs'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'configuration' / 'perturbation'

if __name__ == '__main__':
    range = [0.016, 0.08]
    mean = numpy.mean(range)
    std = mean / 3

    values = numpy.random.normal(mean, std, 5)

    runs = {
        f'mannings_n_{mannings_n:.3}': (mannings_n, 'mannings_n_at_sea_floor')
        for mannings_n in values
    }

    if not (INPUT_DIRECTORY / 'fort.14').exists():
        download_shinnecock_mesh(INPUT_DIRECTORY)

    nems = ModelingSystem(
        start_time=datetime(2008, 8, 23),
        duration=timedelta(days=14.5),
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
        name='nems_mannings_n_perturbation',
        email_address='zachary.burnett@noaa.gov',
        platform=HPC.HERA,
        spinup=timedelta(days=12.5),
    )
    print('done')
