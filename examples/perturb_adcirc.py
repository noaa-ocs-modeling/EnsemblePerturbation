#! /usr/bin/env python
from datetime import datetime, timedelta
from glob import glob
import os
from pathlib import Path
import re
from shutil import copyfile

from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.server import SlurmConfig
from nemspy import ModelingSystem
from nemspy.model import ADCIRC, AtmosphericMesh, WaveMesh
import numpy

from ensemble_perturbation.inputs.adcirc import download_test_configuration
from ensemble_perturbation.utilities import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = repository_root() / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'output'

if __name__ == '__main__':
    if not os.path.exists(INPUT_DIRECTORY):
        os.makedirs(INPUT_DIRECTORY, exist_ok=True)
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    LOGGER.info('perturbing Manning\'s N')

    fort14_filename = INPUT_DIRECTORY / "fort.14"
    fort15_filename = INPUT_DIRECTORY / "fort.15"

    if not fort14_filename.is_file():
        download_test_configuration(INPUT_DIRECTORY)

    # open mesh file
    mesh = AdcircMesh.open(fort14_filename, crs=4326)

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()

    mesh.add_forcing(tidal_forcing)

    start_time = datetime(2020, 6, 1)
    duration = timedelta(days=7)
    interval = timedelta(hours=1)

    nems = ModelingSystem(start_time, duration, interval,
                          atm=AtmosphericMesh('atm.nc'),
                          wav=WaveMesh('wav.nc'),
                          ocn=ADCIRC(10))

    # instantiate AdcircRun object.
    slurm = SlurmConfig(
        account=None,
        ntasks=100,
        run_name='ADCIRC_GAHM_GENERIC',
        partition='development',
        walltime=timedelta(hours=2),
        nodes=100,
        mail_type='all',
        mail_user='zachary.burnett@noaa.gov',
        log_filename='mannings_n_perturbation.log',
        modules=['intel', 'impi', 'netcdf'],
        path_prefix='$HOME/adcirc/build',
        launcher='ibrun'
    )
    driver = AdcircRun(
        mesh=mesh,
        start_date=start_time,
        end_date=start_time + duration,
        spinup_time=timedelta(days=5),
        server_config=slurm
    )
    driver.import_stations(Path(repository_root()) /
                           'examples/data/stations.txt')
    driver.set_elevation_stations_output(timedelta(minutes=6),
                                         spinup=timedelta(minutes=6))
    driver.set_elevation_surface_output(timedelta(minutes=6),
                                        spinup=timedelta(minutes=6))
    driver.set_velocity_stations_output(timedelta(minutes=6),
                                        spinup=timedelta(minutes=6))
    driver.set_velocity_surface_output(timedelta(minutes=6),
                                       spinup=timedelta(minutes=6))
    for mannings_n in numpy.linspace(0.016, 0.08, 5):
        output_directory = OUTPUT_DIRECTORY / f'mannings_n_{mannings_n:.3}'
        LOGGER.info(f'writing config files for Manning\'s N = {mannings_n:.3} '
                    f'to "{output_directory}"')
        driver.mesh.mannings_n_at_sea_floor = numpy.full(
            [len(driver.mesh.coords)], fill_value=mannings_n)
        driver.write(output_directory, overwrite=True)
        nems.write(output_directory, overwrite=True)

    copyfile(repository_root() / 'ensemble_perturbation/inputs/slurm.job',
             OUTPUT_DIRECTORY / 'slurm.job')

    pattern = re.compile(' p*adcirc')
    replacement = ' NEMS.x'
    for job_filename in glob(str(OUTPUT_DIRECTORY / '**' / 'slurm.job'),
                             recursive=True):
        with open(job_filename) as job_file:
            text = job_file.read()
        matched = pattern.search(text)
        if matched:
            LOGGER.info(f'replacing `{matched.group(0)}` with `{replacement}` '
                        f'in "{job_filename}"')
            text = re.sub(pattern, replacement, text)
            with open(job_filename, 'w') as job_file:
                job_file.write(text)

    copyfile(repository_root() / 'ensemble_perturbation/inputs/slurm.job',
             OUTPUT_DIRECTORY / 'slurm.job')

    copyfile(repository_root() / 'ensemble_perturbation/inputs/slurm.job',
             OUTPUT_DIRECTORY / 'slurm.job')

    print('done')
