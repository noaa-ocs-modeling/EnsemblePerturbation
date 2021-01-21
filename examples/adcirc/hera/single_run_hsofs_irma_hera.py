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

from ensemble_perturbation.configuration.adcirc import write_adcirc_configurations
from ensemble_perturbation.configuration.job_script import HPC
from ensemble_perturbation.utilities import repository_root

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = Path("/scratch2/COASTAL/coastal/save/Saeed.Moghimi/setups/nems_inp/hsofs_grid_v1/")
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'configuration' / 'hsofs'

if __name__ == '__main__':
    runs = {f'nems_hsofs_test': (None, None)}

    if not (INPUT_DIRECTORY / 'fort.14').exists():
        raise RuntimeError(f'file not found at {INPUT_DIRECTORY / "fort.14"}')

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()
    wind_forcing = AtmosphericMeshForcing(17, 3600)
    wave_forcing = WaveWatch3DataForcing(5, 3600)

    nems = ModelingSystem(
        start_time=datetime(2017, 9, 5),
        duration=timedelta(days=14.5),
        interval=timedelta(hours=1),
        atm=AtmosphericMeshEntry('/scratch2/COASTAL/coastal/save/Saeed.Moghimi/setups/nems_inp/'
                                 'hsofs_forcings/irm_v1/inp_atmesh/Wind_HWRF_IRMA_Nov2018_ExtendedSmoothT.nc'),
        wav=WaveMeshEntry('/scratch2/COASTAL/coastal/save/Saeed.Moghimi/setups/nems_inp/'
                          'hsofs_forcings/irm_v1/inp_wavdata/ww3.HWRF.NOV2018.2017_sxy.nc'),
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
        name='nems_hsofs_test',
        email_address='zachary.burnett@noaa.gov',
        platform=HPC.HERA,
        spinup=timedelta(days=12.5),
        forcings=[tidal_forcing, wind_forcing, wave_forcing],
    )
