from datetime import datetime, timedelta

from nemspy import ModelingSystem
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry

from ensemble_perturbation.configuration.adcirc import download_shinnecock_mesh, write_adcirc_configurations
from ensemble_perturbation.configuration.job_script import HPC
from ensemble_perturbation.utilities import repository_root

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input' / 'shinnecock'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'configuration' / 'shinnecock'

if __name__ == '__main__':
    runs = {f'nems_shinnecock_test': (None, None)}

    if not (INPUT_DIRECTORY / 'fort.14').exists():
        download_shinnecock_mesh(INPUT_DIRECTORY)

    nems = ModelingSystem(
        start_time=datetime(2012, 10, 22, 6),
        duration=timedelta(days=14.5),
        interval=timedelta(hours=1),
        atm=AtmosphericMeshEntry('../../forcings/shinnecock/wind_atm_fin_ch_time_vec.nc'),
        wav=WaveMeshEntry('../../forcings/shinnecock/ww3.Constant.20151214_sxy_ike_date.nc'),
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
    )
