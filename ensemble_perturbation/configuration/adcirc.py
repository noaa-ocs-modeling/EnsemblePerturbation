from datetime import datetime, timedelta
from glob import glob
import os
from os import PathLike
from pathlib import Path
import re
from shutil import copyfile
import tarfile

from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.server import SlurmConfig
from nemspy import ModelingSystem
from nemspy.model import ADCIRC, AtmosphericMesh, WaveMesh
import numpy
import requests

from ensemble_perturbation.utilities import get_logger, repository_root

LOGGER = get_logger('configuration.adcirc')


def download_test_configuration(directory: str):
    """
    fetch shinnecock inlet test data
    :param directory: local directory
    """

    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        os.makedirs(directory, exist_ok=True)

    url = 'https://www.dropbox.com/s/1wk91r67cacf132/NetCDF_shinnecock_inlet.tar.bz2?dl=1'
    remote_file = requests.get(url, stream=True)
    temporary_filename = directory / 'temp.tar.gz'
    with open(temporary_filename, 'b+w') as local_file:
        local_file.write(remote_file.raw.read())
    with tarfile.open(temporary_filename, 'r:bz2') as local_file:
        local_file.extractall(directory)
    os.remove(temporary_filename)


def write_adcirc_configurations(
        runs: {str: (float, str)}, input_directory: PathLike, output_directory: PathLike
):
    """
    Generate ADCIRC run configuration for given variable values.

    :param runs: dictionary of run name to run value and mesh attribute name
    :param input_directory: path to input data
    :param output_directory: path to store run configuration
    """

    if not isinstance(input_directory, Path):
        input_directory = Path(input_directory)
    if not isinstance(output_directory, Path):
        input_directory = Path(output_directory)

    if not input_directory.exists():
        os.makedirs(input_directory, exist_ok=True)
    if not output_directory.exists():
        os.makedirs(output_directory, exist_ok=True)

    fort14_filename = input_directory / 'fort.14'

    if not fort14_filename.is_file():
        download_test_configuration(input_directory)

    # open mesh file
    mesh = AdcircMesh.open(fort14_filename, crs=4326)

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()

    mesh.add_forcing(tidal_forcing)

    start_time = datetime(2020, 6, 1)
    duration = timedelta(days=7)
    interval = timedelta(hours=1)

    nems = ModelingSystem(
        start_time,
        duration,
        interval,
        atm=AtmosphericMesh('atm.nc'),
        wav=WaveMesh('wav.nc'),
        ocn=ADCIRC(10),
    )

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
        launcher='ibrun',
    )
    driver = AdcircRun(
        mesh=mesh,
        start_date=start_time,
        end_date=start_time + duration,
        spinup_time=timedelta(days=5),
        server_config=slurm,
    )
    driver.import_stations(Path(repository_root()) / 'examples/data/stations.txt')
    driver.set_elevation_stations_output(timedelta(minutes=6), spinup=timedelta(minutes=6))
    driver.set_elevation_surface_output(timedelta(minutes=6), spinup=timedelta(minutes=6))
    driver.set_velocity_stations_output(timedelta(minutes=6), spinup=timedelta(minutes=6))
    driver.set_velocity_surface_output(timedelta(minutes=6), spinup=timedelta(minutes=6))
    for run_name, (value, attribute_name) in runs.items():
        run_directory = output_directory / run_name
        LOGGER.info(f'writing config files for "{run_directory}"')
        if not isinstance(value, numpy.ndarray):
            value = numpy.full([len(driver.mesh.coords)], fill_value=value)
        if not driver.mesh.has_attribute(attribute_name):
            driver.mesh.add_attribute(attribute_name)
        driver.mesh.set_attribute(attribute_name, value)
        driver.write(run_directory, overwrite=True)
        nems.write(run_directory, overwrite=True)

    copyfile(
        repository_root() / 'ensemble_perturbation/configuration/slurm.job',
        output_directory / 'slurm.job',
    )

    pattern = re.compile(' p*adcirc')
    replacement = ' NEMS.x'
    for job_filename in glob(str(output_directory / '**' / 'slurm.job'), recursive=True):
        with open(job_filename) as job_file:
            text = job_file.read()
        matched = pattern.search(text)
        if matched:
            LOGGER.debug(
                f'replacing `{matched.group(0)}` with `{replacement}`' f' in "{job_filename}"'
            )
            text = re.sub(pattern, replacement, text)
            with open(job_filename, 'w') as job_file:
                job_file.write(text)
