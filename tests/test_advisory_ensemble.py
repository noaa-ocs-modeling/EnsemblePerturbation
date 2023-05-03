import pandas.testing

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
    RadiusOfMaximumWinds,
    perturb_tracks,
)
from tests import check_reference_directory, DATA_DIRECTORY


def test_existing_advisory():
    input_directory = DATA_DIRECTORY / 'input' / 'test_existing_advisory'
    output_directory = DATA_DIRECTORY / 'output' / 'test_existing_advisory'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_existing_advisory'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    variables = [CrossTrack, AlongTrack, MaximumSustainedWindSpeed, RadiusOfMaximumWinds]

    perturbations = perturb_tracks(
        perturbations=9,
        directory=output_directory,
        storm=input_directory / 'florence_advisory.22',
        file_deck='a',
        variables=variables,
        sample_from_distribution=True,
        sample_rule='korobov',
        overwrite=True,
        parallel=True,
    )

    check_reference_directory(output_directory, reference_directory)
