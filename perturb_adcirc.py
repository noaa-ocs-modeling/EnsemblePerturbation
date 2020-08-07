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

DATA_DIRECTORY = pathlib.Path(os.path.expandvars('$WORK')) / "data"
INPUT_DIRECTORY = DATA_DIRECTORY / "NetCDF_Shinnecock_Inlet"
FORT14_FILENAME = INPUT_DIRECTORY / "fort.14"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "output"

if __name__ == '__main__':
    if not os.path.exists(INPUT_DIRECTORY):
        os.makedirs(INPUT_DIRECTORY, exist_ok=True)
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # fetch shinnecock inlet test data
    if not FORT14_FILENAME.is_file():
        url = "https://www.dropbox.com/s/1wk91r67cacf132/NetCDF_shinnecock_inlet.tar.bz2?dl=1"
        g = urllib.request.urlopen(url)
        temporary_directory = tempfile.TemporaryDirectory()
        temporary_filename = os.path.join(temporary_directory.name, 'temp.bz2')
        with open(temporary_filename, 'b+w') as f:
            f.write(g.read())
        with tarfile.open(temporary_filename, "r:bz2") as tar:
            tar.extractall(INPUT_DIRECTORY)

    # open mesh file
    mesh = AdcircMesh.open(FORT14_FILENAME, crs=4326)

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()

    mesh.add_forcing(tidal_forcing)

    # instantiate AdcircRun object.
    slurm = SlurmConfig(
        account='NOAA_CSDL_NWI',
        slurm_ntasks=1000,
        run_name='ADCIRC_GAHM_GENERIC',
        partition='development',
        duration=timedelta(hours=8),
        tasks_per_node=15,
        mail_type='all',
        mail_user='zachary.burnett@noaa.gov',
        log_filename='mannings_n_perturbation.log',
        modules=['intel', 'impi', 'netcdf/4.7.2-parallel'],
        path_prefix='$HOME/adcirc/build'
    )
    driver = AdcircRun(
        mesh=mesh,
        start_date=datetime.now(),
        end_date=timedelta(days=7),
        spinup_time=timedelta(days=5),
        server_config=slurm
    )
    driver.set_elevation_stations_output(timedelta(minutes=6), spinup=timedelta(minutes=6))
    for mannings_n in numpy.linspace(0.001, 0.15, 40):
        driver.mesh.mannings_n_at_sea_floor = numpy.full((len(driver.mesh.coords),),
                                                         fill_value=mannings_n)
        driver.write(OUTPUT_DIRECTORY / f'mannings_n_{mannings_n}')
