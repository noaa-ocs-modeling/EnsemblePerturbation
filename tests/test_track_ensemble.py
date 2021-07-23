import os

from adcircpy.forcing.winds.best_track import FileDeck

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
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
        storm='al062018', start_date='20180911', end_date=None, file_deck=FileDeck.b,
    )

    for filename in output_directory.iterdir():
        os.remove(filename)

    # list of variables where perturbation is Gaussian
    gauss_variables = [MaximumSustainedWindSpeed, CrossTrack, AlongTrack]
    for gauss_variable in gauss_variables:
        perturber.write(
            perturbations=[-1.0, 1.0], variables=[gauss_variable], directory=output_directory,
        )

    # list of variables where perturbation is bounded in the range [0,1)
    range_variables = [RadiusOfMaximumWinds]
    for range_variable in range_variables:
        perturber.write(
            perturbations=[0.25, 0.75], variables=[range_variable], directory=output_directory,
        )

    check_reference_directory(output_directory, reference_directory)


def test_multivariate_besttrack_ensemble():
    output_directory = DATA_DIRECTORY / 'output' / 'test_multivariate_besttrack_ensemble'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_multivariate_besttrack_ensemble'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    perturber = VortexPerturber(
        storm='al062018', start_date='20180911', end_date=None, file_deck=FileDeck.b,
    )

    # list of variables where perturbation is Gaussian
    gauss_variables = [MaximumSustainedWindSpeed, CrossTrack, AlongTrack]
    perturber.write(
        perturbations=[
            -1.0,
            {MaximumSustainedWindSpeed: -0.25, CrossTrack: 0.25, 'along_track': 0.75},
            0.75,
        ],
        variables=gauss_variables,
        directory=output_directory,
        overwrite=True,
    )

    check_reference_directory(output_directory, reference_directory)


def test_original_file():
    output_directory = DATA_DIRECTORY / 'output' / 'test_original_file'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_original_file'
    run_1_directory = output_directory / 'run_1'
    run_2_directory = output_directory / 'run_2'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    original_data = open(reference_directory / 'original.22').read()

    gauss_variables = [MaximumSustainedWindSpeed, CrossTrack]
    range_variables = [RadiusOfMaximumWinds]

    perturber = VortexPerturber(storm='al062018', start_date='20180911', end_date=None)

    perturber.write(
        perturbations=[-1.0, 1.0], variables=gauss_variables, directory=run_1_directory,
    )

    assert open(run_1_directory / 'original.22').read() == original_data

    perturber.write(
        perturbations=[-1.0, 1.0], variables=gauss_variables, directory=run_1_directory,
    )

    assert open(run_1_directory / 'original.22').read() == original_data

    perturber.write(
        perturbations=[-1.0, 1.0], variables=gauss_variables, directory=run_2_directory,
    )

    assert open(run_2_directory / 'original.22').read() == original_data

    perturber.write(
        perturbations=[0.25, 0.75], variables=range_variables, directory=run_2_directory,
    )

    assert open(run_2_directory / 'original.22').read() == original_data
