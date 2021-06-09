from ensembleperturbation.perturbation.make_storm_ensemble import AlongTrack, BestTrackPerturber, CrossTrack, \
    MaximumSustainedWindSpeed, \
    RadiusOfMaximumWinds
from tests import DATA_DIRECTORY, check_reference_directory


def test_besttrack_ensemble():
    output_directory = DATA_DIRECTORY / 'output' / 'test_besttrack_ensemble'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_besttrack_ensemble'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    # hardcoding variable list for now
    variables = [
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWinds,
        AlongTrack,
        CrossTrack,
    ]

    perturber = BestTrackPerturber(
        storm='al062018',
        start_date='20180911',
        end_date=None,
    )

    perturber.write(
        number_of_perturbations=3, variables=variables,
        directory=output_directory,
    )

    check_reference_directory(output_directory, reference_directory)
