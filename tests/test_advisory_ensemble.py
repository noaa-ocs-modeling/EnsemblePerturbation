from datetime import datetime
import pandas.testing

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
    RadiusOfMaximumWindsPersistent,
    RadiusOfMaximumWinds,
    perturb_tracks,
    PerturberFeatures,
)
from tests import check_reference_directory, DATA_DIRECTORY


def test_existing_advisory():
    input_directory = DATA_DIRECTORY / 'input' / 'test_existing_advisory'
    output_directory = DATA_DIRECTORY / 'output' / 'test_existing_advisory'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_existing_advisory'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    variables = [
        CrossTrack,
        AlongTrack,
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWindsPersistent,
    ]

    perturbations = perturb_tracks(
        perturbations=9,
        directory=output_directory,
        storm=input_directory / 'florence_advisory_persistentRMW.22',
        file_deck='a',
        variables=variables,
        sample_from_distribution=True,
        sample_rule='korobov',
        overwrite=True,
        parallel=True,
    )

    check_reference_directory(output_directory, reference_directory)


def test_online_advisory():
    output_directory = DATA_DIRECTORY / 'output' / 'test_online_advisory'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_online_advisory'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    variables = [
        CrossTrack,
        AlongTrack,
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWinds,
    ]

    perturbations = perturb_tracks(
        perturbations=9,
        directory=output_directory,
        storm='Ida2021',
        start_date=datetime(2021, 8, 28),
        advisories=['OFCL'],
        file_deck='a',
        variables=variables,
        sample_from_distribution=True,
        sample_rule='korobov',
        overwrite=True,
        parallel=True,
    )

    check_reference_directory(output_directory, reference_directory)


def test_no_isotach_adj_feature():
    output_directory = DATA_DIRECTORY / 'output' / 'test_no_isotach_adj'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_no_isotach_adj'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    variables = [
        CrossTrack,
        AlongTrack,
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWindsPersistent, # No feature is used for persistent error
    ]

    perturbations = perturb_tracks(
        perturbations=9,
        directory=output_directory,
        storm='Ida2021',
        start_date=datetime(2021, 8, 28),
        advisories=['OFCL'],
        file_deck='a',
        variables=variables,
        sample_from_distribution=True,
        sample_rule='korobov',
        overwrite=True,
        parallel=True,
        features=PerturberFeatures.NONE,
    )

    check_reference_directory(output_directory, reference_directory)


def test_no_feature_equals_none():
    output_directory_1 = DATA_DIRECTORY / 'output' / 'test_no_feature_equals_none' / 'none'
    output_directory_2 = DATA_DIRECTORY / 'output' / 'test_no_feature_equals_none' / 'no_feature'

    if not output_directory_1.exists():
        output_directory_1.mkdir(parents=True, exist_ok=True)
    if not output_directory_2.exists():
        output_directory_2.mkdir(parents=True, exist_ok=True)

    variables = [
        CrossTrack,
        AlongTrack,
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWindsPersistent, # No feature is used for persistent error
    ]

    perturbations = perturb_tracks(
        perturbations=9,
        directory=output_directory_1,
        storm='Ida2021',
        start_date=datetime(2021, 8, 28),
        advisories=['OFCL'],
        file_deck='a',
        variables=variables,
        sample_from_distribution=True,
        sample_rule='korobov',
        overwrite=True,
        parallel=True,
        features=None,
    )

    perturbations = perturb_tracks(
        perturbations=9,
        directory=output_directory_2,
        storm='Ida2021',
        start_date=datetime(2021, 8, 28),
        advisories=['OFCL'],
        file_deck='a',
        variables=variables,
        sample_from_distribution=True,
        sample_rule='korobov',
        overwrite=True,
        parallel=True,
        features=PerturberFeatures.NONE,
    )

    check_reference_directory(output_directory_1, output_directory_2)
