from dateutil.parser import parse as parse_date

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
    RadiusOfMaximumWinds,
    VortexPerturber,
)
from ensembleperturbation.tropicalcyclone.atcf import VortexForcing
from tests import check_reference_directory, DATA_DIRECTORY


def test_besttrack_ensemble():
    output_directory = DATA_DIRECTORY / 'output' / 'test_besttrack_ensemble'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_besttrack_ensemble'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    perturber = VortexPerturber(storm='al062018', start_date='20180911', end_date=None)

    # list of variables where perturbation is Gaussian
    gauss_variables = [MaximumSustainedWindSpeed, CrossTrack, AlongTrack]
    perturber.write(
        number_of_perturbations=2,
        variables=gauss_variables,
        directory=output_directory,
        alphas=[-1.0, 1.0],
    )

    # list of variables where perturbation is bounded in the range [0,1)
    range_variables = [RadiusOfMaximumWinds]
    perturber.write(
        number_of_perturbations=2,
        variables=range_variables,
        directory=output_directory,
        alphas=[0.25, 0.75],
    )

    check_reference_directory(output_directory, reference_directory)


def test_vortex_types():
    output_directory = DATA_DIRECTORY / 'output' / 'test_vortex_types'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_vortex_types'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    file_decks = {
        'a': {
            'start_date': parse_date('2018-09-11 06:00'),
            'end_date': None,
            'record_types': ['OFCL', 'HWRF', 'HMON', 'CARQ'],
        },
        'b': {
            'start_date': parse_date('2018-09-11 06:00'),
            'end_date': parse_date('2018-09-18 06:00'),
            'record_types': ['BEST'],
        },
    }

    for file_deck, values in file_decks.items():
        for record_type in values['record_types']:
            cyclone = VortexForcing(
                'al062018',
                start_date=values['start_date'],
                end_date=values['end_date'],
                file_deck=file_deck,
                requested_record_type=record_type,
            )

            cyclone.write(
                output_directory / f'{file_deck}-deck_{record_type}.txt', overwrite=True
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
        number_of_perturbations=2,
        variables=gauss_variables,
        directory=run_1_directory,
        alphas=[-1.0, 1.0],
    )

    assert open(run_1_directory / 'original.22').read() == original_data

    perturber.write(
        number_of_perturbations=2,
        variables=gauss_variables,
        directory=run_1_directory,
        alphas=[-1.0, 1.0],
    )

    assert open(run_1_directory / 'original.22').read() == original_data

    perturber.write(
        number_of_perturbations=2,
        variables=gauss_variables,
        directory=run_2_directory,
        alphas=[-1.0, 1.0],
    )

    assert open(run_2_directory / 'original.22').read() == original_data

    perturber.write(
        number_of_perturbations=2,
        variables=range_variables,
        directory=run_2_directory,
        alphas=[0.25, 0.75],
    )

    assert open(run_2_directory / 'original.22').read() == original_data
