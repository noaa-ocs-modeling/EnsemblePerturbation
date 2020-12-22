from datetime import datetime, timedelta

from nemspy import ModelingSystem
from nemspy.model import ADCIRCEntry, AtmosphericMeshEntry, WaveMeshEntry

from ensemble_perturbation.configuration.adcirc import download_test_configuration, write_adcirc_configurations
from ensemble_perturbation.utilities import repository_root

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input' / 'hsofs'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'configuration' / 'hsofs'

if __name__ == '__main__':
    runs = {f'nems_hsofs_test': (None, None)}

    if not (INPUT_DIRECTORY / 'fort.14').exists():
        download_test_configuration(INPUT_DIRECTORY)

    nems = ModelingSystem(
        start_time=datetime(2012, 10, 22, 6),
        duration=timedelta(days=14.5),
        interval=timedelta(hours=1),
        atm=AtmosphericMeshEntry('../../forcings/hsofs/Wind_HWRF_SANDY_Nov2018_ExtendedSmoothT.nc'),
        wav=WaveMeshEntry('../../forcings/hsofs/ww3.HWRF.NOV2018.2012_sxy.nc'),
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
        tacc=True,
        spinup=timedelta(days=12.5),
    )
