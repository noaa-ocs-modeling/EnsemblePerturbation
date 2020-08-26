#! /usr/bin/env python

from datetime import datetime, timedelta
import os
from pathlib import Path
import tarfile

from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.server import SlurmConfig
import numpy
import requests

from ensemble_perturbation import get_logger, repository_root

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = Path(repository_root()) / 'examples/data'
INPUT_DIRECTORY = DATA_DIRECTORY / 'input'
OUTPUT_DIRECTORY = DATA_DIRECTORY / 'output'


def download_test_configuration(directory: str):
    """
    fetch shinnecock inlet test data
    :param directory: local directory
    """

    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        os.makedirs(directory, exist_ok=True)

    url = "https://www.dropbox.com/s/1wk91r67cacf132/" \
          "NetCDF_shinnecock_inlet.tar.bz2?dl=1"
    remote_file = requests.get(url, stream=True)
    temporary_filename = DATA_DIRECTORY / 'temp.tar.gz'
    with open(temporary_filename, 'b+w') as local_file:
        local_file.write(remote_file.raw.read())
    with tarfile.open(temporary_filename, "r:bz2") as local_file:
        local_file.extractall(directory)
    os.remove(temporary_filename)


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
        start_date=datetime.now(),
        end_date=timedelta(days=7),
        spinup_time=timedelta(days=5),
        server_config=slurm
    )
    driver.import_stations(Path(repository_root()) /
                           'examples/data/stations.txt')
    driver.set_elevation_stations_output(timedelta(minutes=6),
                                         spinup=timedelta(minutes=6))
    driver.set_elevation_surface_output(timedelta(minutes=6),
                                        spinup=timedelta(minutes=6))
    for mannings_n in numpy.linspace(0.016, 0.08, 5):
        output_directory = OUTPUT_DIRECTORY / f'mannings_n_{mannings_n:.3}'
        LOGGER.info(f'writing config files for Manning\'s N = {mannings_n:.3} '
                    f'to "{output_directory}"')
        driver.mesh.mannings_n_at_sea_floor = numpy.full(
            [len(driver.mesh.coords)], fill_value=mannings_n)
        driver.write(output_directory, overwrite=True)

    print('done')
