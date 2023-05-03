import os
from datetime import datetime
import pandas.testing
from stormevents.nhc import VortexTrack
from stormevents.nhc.atcf import ATCF_FIELDS

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
        storm='al062018', start_date=datetime(2018, 9, 11), end_date=None, file_deck='b',
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
        storm='al062018', start_date=datetime(2018, 9, 11), end_date=None, file_deck='b',
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
            file_deck='b',
            sample_rule='random',
            sample_from_distribution=True,
            quadrature=False,
            overwrite=True,
        )

    for i in range(4):
        assert (output_directory / f'vortex_1_variable_random_{i+1}.json').is_file()
        assert (output_directory / f'vortex_1_variable_random_{i+1}.22').is_file()


def test_original_file():
    output_directory = DATA_DIRECTORY / 'output' / 'test_original_file'
    run_1_directory = output_directory / 'run_1'
    run_2_directory = output_directory / 'run_2'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    original_track_0 = VortexTrack(
        storm='al062018', start_date=datetime(2018, 9, 11), end_date=None, file_deck='b',
    )
    original_track_0.to_file(output_directory / 'original.22')

    gauss_variables = [MaximumSustainedWindSpeed, CrossTrack]
    range_variables = [RadiusOfMaximumWinds]

    perturber = VortexPerturber.from_file(
        output_directory / 'original.22', start_date=datetime(2018, 9, 11), file_deck='b',
    )

    perturber.write(
        perturbations=[1.0],
        variables=gauss_variables,
        directory=run_1_directory,
        overwrite=True,
    )
    original_track_1 = VortexTrack.from_file(run_1_directory / 'original.22')

    perturber.write(
        perturbations=[1.0],
        variables=gauss_variables,
        directory=run_2_directory,
        overwrite=True,
    )
    original_track_2 = VortexTrack.from_file(run_2_directory / 'original.22')

    perturber.write(
        perturbations=[1.0],
        variables=range_variables,
        directory=run_2_directory,
        overwrite=True,
    )
    original_track_3 = VortexTrack.from_file(run_2_directory / 'original.22')

    comparison_fields = [
        field
        for field in ATCF_FIELDS.values()
        if field not in ['direction', 'speed', 'extra_values']
    ]
    original_data = original_track_0.data[comparison_fields].reset_index(drop=True)
    pandas.testing.assert_frame_equal(original_track_1.data[comparison_fields], original_data)
    pandas.testing.assert_frame_equal(original_track_2.data[comparison_fields], original_data)
    pandas.testing.assert_frame_equal(original_track_3.data[comparison_fields], original_data)
