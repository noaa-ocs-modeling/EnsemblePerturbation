#! /usr/bin/env python
from datetime import datetime, timedelta
import os
from pathlib import Path
from random import gauss

import click
from coupledmodeldriver import Platform
from coupledmodeldriver.configure import BestTrackForcingJSON, TidalForcingJSON
from coupledmodeldriver.generate import (
    ADCIRCRunConfiguration,
    generate_adcirc_configuration,
    NEMSADCIRCRunConfiguration,
)
import numpy

from ensembleperturbation.perturbation.atcf import AlongTrack, CrossTrack, MaximumSustainedWindSpeed, RadiusOfMaximumWinds, \
    VortexPerturber
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('perturb.adcirc')

OUTPUT_DIRECTORY = Path.cwd() / f'run_{datetime.now():%Y%m%d}_perturbed_track_example'
if not OUTPUT_DIRECTORY.exists():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

SHARED_DIRECTORY = Path('/work/noaa/nosofs/share')
TRACK_DIRECTORY = OUTPUT_DIRECTORY / 'track_files'
if not TRACK_DIRECTORY.exists():
    TRACK_DIRECTORY.mkdir(parents=True, exist_ok=True)
ORIGINAL_TRACK_FILENAME = TRACK_DIRECTORY / 'fort.22'

# start and end times for model
STORM = 'al062018'
MODELED_START_TIME = datetime(year=2018, month=9, day=11, hour=6)
MODELED_DURATION = timedelta(days=6)
MODELED_TIMESTEP = timedelta(seconds=2)
TIDAL_SPINUP_DURATION = timedelta(days=12.5)
NEMS_INTERVAL = timedelta(hours=1)

# directories containing forcings and mesh
MESH_DIRECTORY = (
    SHARED_DIRECTORY / 'models' / 'meshes' / 'hsofs' / '120m' / 'Subsetted_Florence2018_Test'
)
FORCINGS_DIRECTORY = SHARED_DIRECTORY / 'models' / 'forcings' / 'hsofs' / '250m' / 'florence'
HAMTIDE_DIRECTORY = SHARED_DIRECTORY / 'models' / 'forcings' / 'tides' / 'hamtide'
TPXO_FILENAME = SHARED_DIRECTORY / 'models' / 'forcings' / 'tides' / 'h_tpxo9.v1.nc'

# connections between coupled components
NEMS_CONNECTIONS = ['WAV -> OCN']
NEMS_SEQUENCE = [
    'WAV -> OCN',
    'WAV',
    'OCN',
]

# platform-specific parameters
PLATFORM = Platform.ORION
ADCIRC_PROCESSORS = 15 * PLATFORM.value['processors_per_node']
NEMS_EXECUTABLE = (
    SHARED_DIRECTORY / 'repositories' / 'CoastalApp_SCHISM' / 'NEMS' / 'exe' / 'NEMS.x'
)
ADCIRC_EXECUTABLE = (
    SHARED_DIRECTORY / 'repositories' / 'CoastalApp_SCHISM' / 'ADCIRC' / 'work' / 'padcirc'
)
ADCPREP_EXECUTABLE = (
    SHARED_DIRECTORY / 'repositories' / 'CoastalApp_SCHISM' / 'ADCIRC' / 'work' / 'adcprep'
)
ASWIP_EXECUTABLE = (
    SHARED_DIRECTORY / 'repositories' / 'CoastalApp_SCHISM' / 'ADCIRC' / 'work' / 'aswip'
)
MODULEFILE = (
    SHARED_DIRECTORY
    / 'repositories'
    / 'CoastalApp_SCHISM'
    / 'modulefiles'
    / 'envmodules_intel.orion'
)
JOB_DURATION = timedelta(minutes=30)

if __name__ == '__main__':
    forcing_configurations = [TidalForcingJSON(resource=TPXO_FILENAME)]

    if ORIGINAL_TRACK_FILENAME.exists():
        from adcircpy.forcing import BestTrackForcing

        forcing_configurations.append(
            BestTrackForcingJSON.from_adcircpy(
                BestTrackForcing.from_fort22(
                    ORIGINAL_TRACK_FILENAME,
                    start_date=MODELED_START_TIME,
                    end_date=MODELED_START_TIME + MODELED_DURATION,
                )
            )
        )

        perturber = VortexPerturber.from_file(
            ORIGINAL_TRACK_FILENAME,
            start_date=MODELED_START_TIME,
            end_date=MODELED_START_TIME + MODELED_DURATION,
        )
    else:
        forcing_configurations.append(
            BestTrackForcingJSON(
                storm_id=STORM,
                start_date=MODELED_START_TIME,
                end_date=MODELED_START_TIME + MODELED_DURATION,
            )
        )

        perturber = VortexPerturber(
            storm=STORM,
            start_date=MODELED_START_TIME,
            end_date=MODELED_START_TIME + MODELED_DURATION,
        )

    variable_perturbations = {
        CrossTrack: [gauss(0.5, 0.25) for _ in range(10)],
        AlongTrack: [gauss(0.5, 0.25) for _ in range(5)],
        RadiusOfMaximumWinds: numpy.linspace(0.25, 0.75, 3),
        MaximumSustainedWindSpeed: [gauss(0, 1) for _ in range(4)],
    }

    track_filenames = [TRACK_DIRECTORY / 'original.22']
    for variable, alphas in variable_perturbations.items():
        track_filenames += perturber.write(
            number_of_perturbations=len(alphas),
            variables=[variable],
            directory=TRACK_DIRECTORY,
            alphas=alphas,
        )

    perturbations = {
        f'besttrack_{index}': {
            'besttrack': {
                'fort22_filename': Path(os.path.relpath(track_filename, OUTPUT_DIRECTORY))
            }
        }
        for index, track_filename in enumerate(track_filenames)
    }

    if NEMS_INTERVAL is not None:
        configuration = NEMSADCIRCRunConfiguration(
            mesh_directory=MESH_DIRECTORY,
            modeled_start_time=MODELED_START_TIME,
            modeled_end_time=MODELED_START_TIME + MODELED_DURATION,
            modeled_timestep=MODELED_TIMESTEP,
            nems_interval=NEMS_INTERVAL,
            nems_connections=None,
            nems_mediations=None,
            nems_sequence=None,
            tidal_spinup_duration=TIDAL_SPINUP_DURATION,
            platform=PLATFORM,
            perturbations=perturbations,
            forcings=forcing_configurations,
            adcirc_processors=ADCIRC_PROCESSORS,
            slurm_partition=None,
            slurm_job_duration=JOB_DURATION,
            slurm_email_address=None,
            nems_executable=ADCIRC_EXECUTABLE,
            adcprep_executable=ADCPREP_EXECUTABLE,
            aswip_executable=ASWIP_EXECUTABLE,
            source_filename=MODULEFILE,
        )
    else:
        configuration = ADCIRCRunConfiguration(
            mesh_directory=MESH_DIRECTORY,
            modeled_start_time=MODELED_START_TIME,
            modeled_end_time=MODELED_START_TIME + MODELED_DURATION,
            modeled_timestep=MODELED_TIMESTEP,
            tidal_spinup_duration=TIDAL_SPINUP_DURATION,
            platform=PLATFORM,
            perturbations=perturbations,
            forcings=forcing_configurations,
            adcirc_processors=ADCIRC_PROCESSORS,
            slurm_partition=None,
            slurm_job_duration=JOB_DURATION,
            slurm_email_address=None,
            adcirc_executable=ADCIRC_EXECUTABLE,
            adcprep_executable=ADCPREP_EXECUTABLE,
            aswip_executable=ASWIP_EXECUTABLE,
            source_filename=MODULEFILE,
        )

    configuration.write_directory(OUTPUT_DIRECTORY, overwrite=True)

    if click.confirm('generate configuration?', default=True):
        os.chdir(OUTPUT_DIRECTORY)
        generate_adcirc_configuration(OUTPUT_DIRECTORY, overwrite=True)

    print('done')
