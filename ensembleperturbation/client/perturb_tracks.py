from pathlib import Path
from argparse import ArgumentParser

from coupledmodeldriver.client.initialize_adcirc import (
    initialize_adcirc,
    parse_initialize_adcirc_arguments,
)
from coupledmodeldriver.client.initialize_schism import (
    initialize_schism,
    parse_initialize_schism_arguments,
)
from coupledmodeldriver.configure import BestTrackForcingJSON

from ensembleperturbation.perturbation.atcf import perturb_tracks


def main():

    # Using a preliminary parser to decide whether to create
    # "adcirc" or "schism" parser below (args of this parser need
    # to be passed as extra_arguments to the subsequent parser)
    preliminary_parser = ArgumentParser(add_help=False)
    preliminary_parser.add_argument(
        '--adcirc', action='store_true',
    )
    preliminary_parser.add_argument(
        '--schism', action='store_true',
    )
    prelim_args, _ = preliminary_parser.parse_known_args()

    init_schism = prelim_args.schism
    init_adcirc = prelim_args.adcirc or not init_schism

    if init_adcirc:
        parse_func = parse_initialize_adcirc_arguments
        init_func = initialize_adcirc
    elif init_schism:
        parse_func = parse_initialize_schism_arguments
        init_func = initialize_schism

    arguments = parse_func(
        extra_arguments={
            'adcirc': (bool, 'Initialize ADCIRC simulation ensemble (default)'),
            'schism': (bool, 'Initialize SCHISM simulation ensemble'),
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

    ocean_model_specific_args = {}
    if init_adcirc:
        ocean_model_specific_args = dict(
            adcirc_executable=arguments['adcirc_executable'],
            adcprep_executable=arguments['adcprep_executable'],
            aswip_executable=arguments['aswip_executable'],
            adcirc_processors=arguments['adcirc_processors'],
        )

    elif init_schism:
        ocean_model_specific_args = dict(
            schism_executable=arguments['schism_executable'],
            schism_processors=arguments['schism_processors'],
            schism_hotstart_combiner=arguments['schism_hotstart_combiner'],
            schism_schout_combiner=arguments['schism_schout_combiner'],
            schism_use_old_io=arguments['schism_use_old_io'],
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
                storm = forcing.pyschism_forcing.nhc_code
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

    init_func(
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
        job_duration=arguments['job_duration'],
        output_directory=arguments['output_directory'],
        absolute_paths=arguments['absolute_paths'],
        overwrite=arguments['overwrite'],
        verbose=arguments['verbose'],
        **ocean_model_specific_args,
    )


if __name__ == '__main__':
    main()
