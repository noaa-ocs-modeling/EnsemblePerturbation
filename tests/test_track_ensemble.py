import os

import numpy
from stormevents import VortexTrack
from stormevents.nhc.atcf import ATCF_FileDeck

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
    perturb_tracks,
    RadiusOfMaximumWinds,
    VortexPerturber,
)
from tests import check_reference_directory, DATA_DIRECTORY


def test_monovariate_besttrack_ensemble():
    output_directory = DATA_DIRECTORY / 'output' / 'test_monovariate_besttrack_ensemble'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_monovariate_besttrack_ensemble'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    perturber = VortexPerturber(
        storm='al062018', start_date='20180911', end_date=None, file_deck=ATCF_FileDeck.b,
    )

    for filename in output_directory.iterdir():
        os.remove(filename)

    # list of variables to perturb
    variables = [MaximumSustainedWindSpeed, CrossTrack, AlongTrack, RadiusOfMaximumWinds]

    # perturb variables one at a time
    for variable in variables:
        perturber.write(
            perturbations=[-1.0, 1.0],
            variables=[variable],
            directory=output_directory,
            continue_numbering=True,
        )

    check_reference_directory(output_directory, reference_directory)


def test_multivariate_besttrack_ensemble():
    output_directory = DATA_DIRECTORY / 'output' / 'test_multivariate_besttrack_ensemble'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_multivariate_besttrack_ensemble'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    perturber = VortexPerturber(
        storm='al062018', start_date='20180911', end_date=None, file_deck=ATCF_FileDeck.b,
    )

    # list of variables to perturb
    variables = [MaximumSustainedWindSpeed, CrossTrack, AlongTrack, RadiusOfMaximumWinds]

    # perturb all variables at once
    perturber.write(
        perturbations=[
            -1.0,
            {
                MaximumSustainedWindSpeed: -0.25,
                CrossTrack: 0.25,
                'along_track': 0.75,
                'radius_of_maximum_winds': -1,
            },
            0.75,
        ],
        variables=variables,
        directory=output_directory,
        quadrature=False,
        overwrite=True,
        parallel=True,
    )

    check_reference_directory(output_directory, reference_directory)


def test_spatial_perturbations():
    output_directory = DATA_DIRECTORY / 'output' / 'test_spatial_perturbations'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    # list of spatial perturbations
    variables = [CrossTrack, AlongTrack]

    unchanged_perturbations = []
    for variable in variables:
        perturbations = perturb_tracks(
            perturbations=4,
            directory=output_directory,
            storm='florence2018',
            variables=[variable],
            sample_from_distribution=True,
            quadrature=False,
            overwrite=True,
        )

        tracks = {
            name: VortexTrack.from_fort22(
                output_directory.parent / perturbation['besttrack']['fort22_filename']
            )
            for name, perturbation in perturbations.items()
        }

        original_track = tracks['original']
        del tracks['original']

        for run, track in tracks.items():
            same = numpy.allclose(
                track.data[['longitude', 'latitude']],
                original_track.data[['longitude', 'latitude']],
            )

            if same:
                unchanged_perturbations.append(variable.name)
                break

    assert (
        len(unchanged_perturbations) == 0
    ), f'failure in {unchanged_perturbations} track perturbation'


def test_original_file():
    output_directory = DATA_DIRECTORY / 'output' / 'test_original_file'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_original_file'
    run_1_directory = output_directory / 'run_1'
    run_2_directory = output_directory / 'run_2'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    reference_track = VortexTrack.from_fort22(reference_directory / 'original.22')

    gauss_variables = [MaximumSustainedWindSpeed, CrossTrack]
    range_variables = [RadiusOfMaximumWinds]

    perturber = VortexPerturber(storm='al062018', start_date='20180911', end_date=None)

    perturber.write(
        perturbations=[-1.0, 1.0], variables=gauss_variables, directory=run_1_directory
    )

    assert VortexTrack.from_fort22(run_1_directory / 'original.22') == reference_track

    perturber.write(
        perturbations=[-1.0, 1.0], variables=gauss_variables, directory=run_1_directory
    )

    assert VortexTrack.from_fort22(run_1_directory / 'original.22') == reference_track

    perturber.write(
        perturbations=[-1.0, 1.0], variables=gauss_variables, directory=run_2_directory
    )

    assert VortexTrack.from_fort22(run_2_directory / 'original.22') == reference_track

    perturber.write(
        perturbations=[-1.0, 1.0], variables=range_variables, directory=run_2_directory
    )

    assert VortexTrack.from_fort22(run_2_directory / 'original.22') == reference_track
