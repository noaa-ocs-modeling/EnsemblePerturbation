#! /usr/bin/env python
from datetime import datetime, timedelta
import os
import pathlib
import tarfile
import tempfile
import urllib.request

from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.server import SlurmConfig
import numpy

from ensemble_perturbation import get_logger

LOGGER = get_logger('perturb.adcirc')

DATA_DIRECTORY = pathlib.Path(os.path.expanduser('~\Downloads')) / "data"
INPUT_DIRECTORY = DATA_DIRECTORY / "Shinnecock_Inlet_NetCDF_output"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "output"

if __name__ == '__main__':
    if not os.path.exists(INPUT_DIRECTORY):
        os.makedirs(INPUT_DIRECTORY, exist_ok=True)
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    LOGGER.info('perturbing Manning\'s N')

    fort14_filename = INPUT_DIRECTORY / "fort.14"
    fort15_filename = INPUT_DIRECTORY / "fort.15"

    # fetch shinnecock inlet test data
    if not fort14_filename.is_file():
        url = "https://www.dropbox.com/s/1wk91r67cacf132/NetCDF_shinnecock_inlet.tar.bz2?dl=1"
        remote_file = urllib.request.urlopen(url)
        temporary_directory = tempfile.TemporaryDirectory()
        temporary_filename = os.path.join(temporary_directory.name, 'temp.bz2')
        with open(temporary_filename, 'b+w') as local_file:
            local_file.write(remote_file.read())
        with tarfile.open(temporary_filename, "r:bz2") as local_file:
            local_file.extractall(INPUT_DIRECTORY)

    # open mesh file
    mesh = AdcircMesh.open(fort14_filename, crs=4326)

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()

    mesh.add_forcing(tidal_forcing)

    # instantiate AdcircRun object.
    slurm = SlurmConfig(
        account=None,
        ntasks=10,
        run_name='ADCIRC_GAHM_GENERIC',
        partition='development',
        walltime=timedelta(hours=2),
        nodes=10,
        mail_type='all',
        mail_user='zachary.burnett@noaa.gov',
        log_filename='mannings_n_perturbation.log',
        modules=['intel', 'impi', 'netcdf'],
        path_prefix='$HOME/adcirc/build'
    )
    driver = AdcircRun(
        mesh=mesh,
        start_date=datetime.now(),
        end_date=timedelta(days=7),
        spinup_time=timedelta(days=5),
        server_config=slurm
    )
    driver.set_elevation_stations_output(timedelta(minutes=6),
                                         spinup=timedelta(minutes=6))
    driver.set_elevation_surface_output(timedelta(minutes=6),
                                        spinup=timedelta(minutes=6))
    driver.import_stations(fort15=fort15_filename)
    for mannings_n in numpy.linspace(0.001, 0.15, 40):
        output_directory = OUTPUT_DIRECTORY / f'mannings_n_{mannings_n:.3}'
        LOGGER.info(f'writing config files for Manning\'s N = {mannings_n:.3} '
                    f'to "{output_directory}"')
        driver.mesh.mannings_n_at_sea_floor = numpy.full(
            [len(driver.mesh.coords)], fill_value=mannings_n)
        driver.write(output_directory, overwrite=True)
