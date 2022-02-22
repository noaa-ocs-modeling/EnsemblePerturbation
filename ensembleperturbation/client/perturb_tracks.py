from pathlib import Path

from coupledmodeldriver.client.initialize_adcirc import (
    initialize_adcirc,
    parse_initialize_adcirc_arguments,
)
from coupledmodeldriver.configure import BestTrackForcingJSON

from ensembleperturbation.perturbation.atcf import perturb_tracks


def main():
    arguments = parse_initialize_adcirc_arguments(
        extra_arguments={
            'perturbations': (int, 'number of perturbations to create'),
            'variables': ([str], 'vortex variables to perturb'),
            'sample': (
                bool,
                'override given perturbations with random samples from the joint distribution',
            ),
            'sample-rule': (
                str,
                "rule to use for the distribution sampling. Please choose from: 'random' [default], 'sobol', 'halton', 'hammersley', 'korobov', 'additive_recursion', or 'latin_hypercube'",
            ),
            'quadrature': (bool, 'add additional perturbations along the quadrature'),
        }
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
    original_track_filename = track_directory / 'original.22'

    if original_track_filename.exists():
        storm = original_track_filename

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
                storm = forcing.adcircpy_forcing.nhc_code
                break
        else:
            raise ValueError('no best track forcing specified')

    perturbations = perturb_tracks(
        perturbations=arguments['perturbations'],
        directory=Path(arguments['output_directory']) / 'track_files',
        storm=storm,
        variables=arguments['variables'],
        sample_from_distribution=arguments['sample'],
        sample_rule=arguments['sample-rule'],
        quadrature=arguments['quadrature'],
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
