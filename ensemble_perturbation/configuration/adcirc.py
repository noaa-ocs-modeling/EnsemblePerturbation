import copy
from datetime import timedelta
from glob import glob
import os
from os import PathLike
from pathlib import Path
import re
import tarfile

from adcircpy import AdcircMesh, AdcircRun, Tides
from adcircpy.forcing.waves import WaveForcing
from adcircpy.forcing.winds import WindForcing
from adcircpy.server import SlurmConfig
from nemspy import ModelingSystem
from nemspy.model import ADCIRCEntry
import numpy
import requests

from .job_script import EnsembleSlurmScript, HPC, SlurmEmailType
from ..utilities import get_logger, repository_root

LOGGER = get_logger('configuration.adcirc')


def write_adcirc_configurations(
    nems: ModelingSystem,
    runs: {str: (float, str)},
    input_directory: PathLike,
    output_directory: PathLike,
    name: str = None,
    partition: str = None,
    email_address: str = None,
    tacc: bool = False,
    wall_clock_time: timedelta = None,
    spinup: timedelta = None,
    source_filename: PathLike = None,
):
    """
    Generate ADCIRC run configuration for given variable values.

    :param runs: dictionary of run name to run value and mesh attribute name
    :param nems: NEMSpy ModelingSystem object, populated with models and connections
    :param input_directory: path to input data
    :param output_directory: path to store run configuration
    :param name: name of this perturbation
    :param partition: Slurm partition
    :param email_address: email address
    :param tacc: whether to configure for TACC
    :param wall_clock_time: wall clock time of job
    :param spinup: spinup time for ADCIRC coldstart
    """

    if not isinstance(input_directory, Path):
        input_directory = Path(input_directory)
    if not isinstance(output_directory, Path):
        input_directory = Path(output_directory)

    if not input_directory.exists():
        os.makedirs(input_directory, exist_ok=True)
    if not output_directory.exists():
        os.makedirs(output_directory, exist_ok=True)

    if name is None:
        name = 'nems_run'

    if 'ocn' not in nems or not isinstance(nems['ocn'], ADCIRCEntry):
        nems['ocn'] = ADCIRCEntry(11)

    fort14_filename = input_directory / 'fort.14'

    launcher = 'ibrun' if tacc else 'srun'
    run_name = 'ADCIRC_GAHM_GENERIC'

    if partition is None:
        partition = 'development'

    if wall_clock_time is None:
        wall_clock_time = timedelta(minutes=30)

    if not fort14_filename.is_file():
        raise FileNotFoundError(f'no input file at {fort14_filename}')

    if spinup is not None and isinstance(spinup, timedelta):
        spinup = ModelingSystem(
            nems.start_time - spinup,
            spinup,
            nems.interval,
            ocn=copy.deepcopy(nems['OCN']),
            **nems.attributes,
        )

    # open mesh file
    mesh = AdcircMesh.open(fort14_filename, crs=4326)

    # init tidal forcing and setup requests
    tidal_forcing = Tides()
    tidal_forcing.use_all()
    wind_forcing = WindForcing(17, 3600)
    wave_forcing = WaveForcing(5, 3600)

    mesh.add_forcing(tidal_forcing)
    mesh.add_forcing(wind_forcing)
    mesh.add_forcing(wave_forcing)

    mesh.generate_tau0()

    if source_filename is None:
        source_filename = f'/scratch2/COASTAL/coastal/save/Zachary.Burnett/nems/ADC-WW3-NWM-NEMS/modulefiles/{"stampede" if tacc else "hera"}/ESMF_NUOPC'

    atm_namelist_filename = output_directory / 'atm_namelist.rc'

    if spinup is None:
        coldstart_filenames = nems.write(
            output_directory, overwrite=True, include_version=True
        )
    else:
        coldstart_filenames = spinup.write(
            output_directory, overwrite=True, include_version=True
        )

    for filename in coldstart_filenames + [atm_namelist_filename]:
        coldstart_filename = Path(f'{filename}.coldstart')
        if coldstart_filename.exists():
            os.remove(coldstart_filename)
        filename.rename(coldstart_filename)

    if spinup is not None:
        hotstart_filenames = nems.write(output_directory, overwrite=True, include_version=True)
    else:
        hotstart_filenames = []

    for filename in hotstart_filenames + [atm_namelist_filename]:
        hotstart_filename = Path(f'{filename}.hotstart')
        if hotstart_filename.exists():
            os.remove(hotstart_filename)
        filename.rename(hotstart_filename)

    ensemble_slurm_script = EnsembleSlurmScript(
        account=None,
        tasks=nems.processors,
        duration=wall_clock_time,
        partition=partition,
        hpc=HPC.TACC if tacc else HPC.ORION,
        launcher=launcher,
        run=name,
        email_type=SlurmEmailType.ALL if email_address is not None else None,
        email_address=email_address,
        error_filename=f'{name}.err.log',
        log_filename=f'{name}.out.log',
        modules=[],
        commands=[f'source {source_filename}'],
    )
    ensemble_slurm_script.write(output_directory, overwrite=True)

    slurm = SlurmConfig(
        account=None,
        ntasks=nems.processors,
        run_name=run_name,
        partition=partition,
        walltime=wall_clock_time,
        nodes=int(numpy.ceil(nems.processors / 68)) if tacc else None,
        mail_type='all' if email_address is not None else None,
        mail_user=email_address,
        log_filename=f'{name}.out.log',
        modules=[],
        launcher=launcher,
        extra_commands=[f'source {source_filename}'],
    )

    # instantiate AdcircRun object.
    driver = AdcircRun(
        mesh=mesh,
        start_date=nems.start_time,
        end_date=nems.start_time + nems.duration,
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
        LOGGER.info(f'writing config to "{run_directory}"')
        if not isinstance(value, numpy.ndarray):
            value = numpy.full([len(driver.mesh.coords)], fill_value=value)
        if not driver.mesh.has_attribute(attribute_name):
            driver.mesh.add_attribute(attribute_name)
        driver.mesh.set_attribute(attribute_name, value)
        driver.write(run_directory, overwrite=True)
        for phase in ['coldstart', 'hotstart']:
            directory = run_directory / phase
            if not directory.exists():
                directory.mkdir()

    atm_namelist_filename = output_directory / 'atm_namelist.rc'

    if spinup is None:
        coldstart_filenames = nems.write(
            output_directory, overwrite=True, include_version=True
        )
    else:
        coldstart_filenames = spinup.write(
            output_directory, overwrite=True, include_version=True
        )

    for filename in coldstart_filenames + [atm_namelist_filename]:
        coldstart_filename = Path(f'{filename}.coldstart')
        if coldstart_filename.exists():
            os.remove(coldstart_filename)
        filename.rename(coldstart_filename)

    if spinup is not None:
        hotstart_filenames = nems.write(output_directory, overwrite=True, include_version=True)
    else:
        hotstart_filenames = []

    for filename in hotstart_filenames + [atm_namelist_filename]:
        hotstart_filename = Path(f'{filename}.hotstart')
        if hotstart_filename.exists():
            os.remove(hotstart_filename)
        filename.rename(hotstart_filename)

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
