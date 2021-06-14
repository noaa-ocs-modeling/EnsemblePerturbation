from ensembleperturbation.perturbation.make_storm_ensemble import (
    AlongTrack,
    VortexPerturber,
    CrossTrack,
    MaximumSustainedWindSpeed,
    RadiusOfMaximumWinds,
)
from tests import check_reference_directory, DATA_DIRECTORY


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

    perturber = VortexPerturber(storm='al062018', start_date='20180911', end_date=None)

    perturber.write(
        number_of_perturbations=3, variables=variables, directory=output_directory, alpha=0.5,
    )

    check_reference_directory(output_directory, reference_directory)
