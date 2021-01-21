#! /usr/bin/env python

from datetime import datetime, timedelta
from pathlib import Path
import sys

from adcircpy import Tides
from adcircpy.forcing.waves.ww3 import WaveWatch3DataForcing
from adcircpy.forcing.winds.atmesh import AtmosphericMeshForcing
from nemspy import ModelingSystem
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from ensemble_perturbation.configuration.adcirc import download_shinnecock_mesh, write_adcirc_configurations
from ensemble_perturbation.configuration.job_script import HPC
from ensemble_perturbation.utilities import repository_root

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input' / 'stampede2'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'configuration' / 'stampede2'

if __name__ == '__main__':
    runs = {f'nems_shinnecock_test': (None, None)}

    if not (INPUT_DIRECTORY / 'fort.14').exists():
        download_shinnecock_mesh(INPUT_DIRECTORY)

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()
    wind_forcing = AtmosphericMeshForcing(17, 3600)
    wave_forcing = WaveWatch3DataForcing(5, 3600)

    nems = ModelingSystem(
        start_time=datetime(2008, 8, 23),
        duration=timedelta(days=14.5),
        interval=timedelta(hours=1),
        atm=AtmosphericMeshEntry('/scratch2/COASTAL/coastal/save/Zachary.Burnett/forcings/'
                                 'shinnecock/ike/wind_atm_fin_ch_time_vec.nc'),
        wav=WaveMeshEntry('/scratch2/COASTAL/coastal/save/Zachary.Burnett/forcings/'
                          'shinnecock/ike/ww3.Constant.20151214_sxy_ike_date.nc'),
        ocn=ADCIRCEntry(382),
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
        name='nems_shinnecock_test',
        email_address='zachary.burnett@noaa.gov',
        platform=HPC.HERA,
        spinup=timedelta(days=12.5),
        forcings=[tidal_forcing, wind_forcing, wave_forcing],
    )
