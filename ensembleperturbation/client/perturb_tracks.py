from pathlib import Path

from coupledmodeldriver.client.initialize_adcirc import (
    initialize_adcirc,
    parse_initialize_adcirc_arguments,
)
from coupledmodeldriver.configure import BestTrackForcingJSON

from ensembleperturbation.perturbation.atcf import perturb_tracks


def main():
    arguments = parse_initialize_adcirc_arguments(
        extra_arguments={'perturbations': int, 'variables': [str]}
    )

    if arguments['perturbations'] is None:
        raise ValueError('number of perturbations not given')

    if arguments['variables'] is None or len(arguments['variables']) == 0:
        arguments['variables'] = [
            'cross_track',
            'along_track',
            'radius_of_maximum_winds',
            'max_sustained_wind_speed',
        ]

    track_directory = arguments['output_directory'] / 'track_files'
    if not track_directory.exists():
        track_directory.mkdir(parents=True, exist_ok=True)
    original_track_filename = track_directory / 'fort.22'

    if original_track_filename.exists():
        storm_id = original_track_filename

        vortex_forcing = BestTrackForcingJSON.from_fort22(
            original_track_filename,
            start_date=arguments['modeled_start_time'],
            end_date=arguments['modeled_start_time'] + arguments['modeled_duration'],
        )
        arguments['forcings'].append(vortex_forcing)
        for index, forcing in enumerate(arguments['forcings']):
            if isinstance(forcing, BestTrackForcingJSON):
                arguments['forcings'][index] = vortex_forcing
                break
    else:
        for forcing in arguments['forcings']:
            if isinstance(forcing, BestTrackForcingJSON):
                storm_id = forcing.adcircpy_forcing.storm_id
                break
        else:
            raise ValueError('no best track forcing specified')

    perturbations = perturb_tracks(
        perturbations=arguments['perturbations'],
        directory=Path(arguments['output_directory']) / 'track_files',
        storm=storm_id,
        variables=arguments['variables'],
        start_date=arguments['modeled_start_time'],
        end_date=arguments['modeled_start_time'] + arguments['modeled_duration'],
        overwrite=arguments['overwrite'],
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
