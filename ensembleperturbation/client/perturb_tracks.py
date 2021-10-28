from datetime import datetime, timedelta
import os
from os import PathLike
from pathlib import Path

from adcircpy.forcing.winds.best_track import FileDeck
from coupledmodeldriver.client.initialize_adcirc import (
    initialize_adcirc,
    parse_initialize_adcirc_arguments,
)
from coupledmodeldriver.configure import BestTrackForcingJSON

from ensembleperturbation.perturbation.atcf import VortexPerturber


def write_vortex_perturbations(
    perturbations: int,
    variables: [str],
    output_directory: PathLike,
    modeled_start_time: datetime,
    modeled_duration: timedelta,
    forcings: [str],
    along_quadrature: bool = False,
    sample_from_distribution: bool = False,
    overwrite: bool = False,
    parallel: bool = False,
):
    if perturbations is None:
        raise ValueError('number of perturbations not given')

    if variables is None or len(variables) == 0:
        variables = [
            'cross_track',
            'along_track',
            'radius_of_maximum_winds',
            'max_sustained_wind_speed',
        ]

    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    track_directory = output_directory / 'track_files'
    if not track_directory.exists():
        track_directory.mkdir(parents=True, exist_ok=True)
    original_track_filename = track_directory / 'original.22'

    if original_track_filename.exists():
        vortex_forcing = BestTrackForcingJSON.from_fort22(
            original_track_filename,
            start_date=modeled_start_time,
            end_date=modeled_start_time + modeled_duration,
        )
        forcings.append(vortex_forcing)
        for index, forcing in enumerate(forcings):
            if isinstance(forcing, BestTrackForcingJSON):
                forcings[index] = vortex_forcing
                break

        perturber = VortexPerturber.from_file(
            original_track_filename,
            start_date=modeled_start_time,
            end_date=modeled_start_time + modeled_duration,
        )
    else:
        for forcing in forcings:
            if isinstance(forcing, BestTrackForcingJSON):
                storm_id = forcing.adcircpy_forcing.storm_id
                break
        else:
            raise ValueError('no best track forcing specified')

        perturber = VortexPerturber(
            storm=storm_id,
            start_date=modeled_start_time,
            end_date=modeled_start_time + modeled_duration,
            file_deck=FileDeck.b,
        )

    track_filenames = [track_directory / 'original.22']
    track_filenames += perturber.write(
        perturbations=perturbations,
        variables=variables,
        directory=track_directory,
        sample_from_distribution=sample_from_distribution,
        quadrature=along_quadrature,
        overwrite=overwrite,
        continue_numbering=False,
        parallel=parallel,
    )

    perturbations = {
        track_filename.stem: {
            'besttrack': {
                'fort22_filename': Path(os.path.relpath(track_filename, output_directory))
            }
        }
        for index, track_filename in enumerate(track_filenames)
    }

    return perturbations


def main():
    arguments = parse_initialize_adcirc_arguments(
        extra_arguments={
            'perturbations': int,
            'quadrature': bool,
            'sample': bool,
            'variables': [str],
            'serial': bool,
        }
    )

    perturbations = write_vortex_perturbations(
        perturbations=arguments['perturbations'],
        variables=arguments['variables'],
        output_directory=arguments['output_directory'],
        modeled_start_time=arguments['modeled_start_time'],
        modeled_duration=arguments['modeled_duration'],
        forcings=arguments['forcings'],
        along_quadrature=arguments['quadrature'],
        sample_from_distribution=arguments['sample'],
        overwrite=arguments['overwrite'],
        parallel=not arguments['serial'],
    )

    initialize_adcirc(
        platform=arguments['platform'],
        mesh_directory=arguments['mesh_directory'],
        modeled_start_time=arguments['modeled_start_time'],
        modeled_duration=arguments['modeled_duration'],
        modeled_timestep=arguments['modeled_timestep'],
        tidal_spinup_duration=arguments['tidal_spinup_duration'],
        perturbations=perturbations,
        nems_connections=None,
        nems_mediations=None,
        nems_sequence=None,
        nems_interval=arguments['nems_interval'],
        modulefile=arguments['modulefile'],
        forcings=arguments['forcings'],
        adcirc_executable=arguments['adcirc_executable'],
        adcprep_executable=arguments['adcprep_executable'],
        aswip_executable=arguments['aswip_executable'],
        adcirc_processors=arguments['adcirc_processors'],
        job_duration=arguments['job_duration'],
        output_directory=arguments['output_directory'],
        absolute_paths=arguments['absolute_paths'],
        overwrite=arguments['overwrite'],
        verbose=arguments['verbose'],
    )


if __name__ == '__main__':
    main()
