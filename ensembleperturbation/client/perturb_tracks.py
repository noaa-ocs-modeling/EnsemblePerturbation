from datetime import datetime, timedelta
import os
from os import PathLike
from pathlib import Path

from adcircpy.forcing.winds.best_track import FileDeck
import chaospy
from coupledmodeldriver.client.initialize_adcirc import (
    initialize_adcirc,
    parse_initialize_adcirc_arguments,
)
from coupledmodeldriver.configure import BestTrackForcingJSON
import numpy

from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable, VortexPerturber

PERTURBED_VARIABLES = {
    variable_class.name: variable_class()
    for variable_class in VortexPerturbedVariable.__subclasses__()
}


def quadrature_perturbations(
    variables: [str], maximum: int = None
) -> (numpy.ndarray, numpy.ndarray):
    """
    Generate quadrature from variable distributions.

    :param variables: names of perturbed variables
    :returns: array of nodes with size NxV, array of weights with size N
    """

    if variables is None or len(variables) == 0:
        variables = [
            'cross_track',
            'along_track',
            'radius_of_maximum_winds',
            'max_sustained_wind_speed',
        ]

    distribution = chaospy.J(
        *(
            PERTURBED_VARIABLES[variable_name].chaospy_distribution()
            for variable_name in variables
        )
    )

    nodes, weights = chaospy.generate_quadrature(
        order=3, dist=distribution, rule='Gaussian', n_max=maximum
    )

    perturbations = [
        {variable: node[index] for index, variable in enumerate(variables)} for node in nodes.T
    ]

    return perturbations, weights


def write_vortex_perturbations(
    perturbations: int,
    variables: [str],
    output_directory: PathLike,
    modeled_start_time: datetime,
    modeled_duration: timedelta,
    forcings: [str],
    overwrite: bool = False,
    parallel: bool = False,
):
    if perturbations is None:
        raise ValueError('number of perturbations not given')

    if variables is None:
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
    original_track_filename = track_directory / 'fort.22'

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
        overwrite=overwrite,
        continue_numbering=True,
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
            'variables': [str],
            'serial': bool,
        }
    )

    if arguments['quadrature']:
        arguments['perturbations'], weights = quadrature_perturbations(
            variables=arguments['variables'], maximum=arguments['perturbations'],
        )
        numpy.save(arguments['output_directory'] / 'weights.npy', weights)

    perturbations = write_vortex_perturbations(
        perturbations=arguments['perturbations'],
        variables=arguments['variables'],
        output_directory=arguments['output_directory'],
        modeled_start_time=arguments['modeled_start_time'],
        modeled_duration=arguments['modeled_duration'],
        forcings=arguments['forcings'],
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
