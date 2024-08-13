import os
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from stormevents.nhc import VortexTrack
from stormevents.nhc.atcf import ATCF_FIELDS

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    perturb_tracks,
    VortexPerturber,
)
from tests import check_reference_directory, DATA_DIRECTORY


def test_track_perturber_forecast_time_init():
    perturber = VortexPerturber(
        storm='al062018', start_date=datetime(2018, 9, 12), end_date=None, file_deck='a',
    )
    dates = np.array(perturber.track.data.track_start_time.unique())
    assert len(dates) == 1
    assert pd.to_datetime(dates.item()) == datetime(2018, 9, 12)


def test_track_perturber_forecast_time_no_startdate():
    perturber = VortexPerturber(storm='al062018', end_date=None, file_deck='a',)
    dates = perturber.track.data.track_start_time.unique()
    assert len(dates) == 1


def test_track_perturber_forecast_time_fromfile():
    original_track = VortexTrack(
        storm='al062018',
        start_date=datetime(2018, 9, 11),
        end_date=None,
        advisories=['OFCL'],
        file_deck='a',
    )

    with tempfile.NamedTemporaryFile(suffix='.22') as fo:
        original_track.to_file(fo.name, overwrite=True)

        perturber = VortexPerturber.from_file(
            Path(fo.name), start_date=datetime(2018, 9, 12), file_deck='a',
        )

    dates = np.array(perturber.track.data.track_start_time.unique())
    assert len(dates) == 1
    assert pd.to_datetime(dates.item()) == datetime(2018, 9, 12)


def test_track_perturber_forecast_time_fromtrack():
    original_track = VortexTrack(
        storm='al062018',
        start_date=datetime(2018, 9, 11),
        end_date=None,
        file_deck='a',
        forecast_time=datetime(2018, 9, 12),
    )

    perturber = VortexPerturber.from_track(original_track,)
    assert perturber.start_date == datetime(2018, 9, 12)

    dates = np.array(perturber.track.data.track_start_time.unique())
    assert len(dates) == 1
    assert pd.to_datetime(dates.item()) == datetime(2018, 9, 12)


def test_perturb_tracks_func_forecast_time():
    output_directory = DATA_DIRECTORY / 'output' / 'test_perturb_tracks_func_forecast_time'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    # list of spatial perturbations
    variables = [CrossTrack, AlongTrack]

    unchanged_perturbations = []
    for variable in variables:
        perturbations = perturb_tracks(
            perturbations=5,
            directory=output_directory,
            storm='florence2018',
            variables=[variable],
            advisories=['OFCL'],
            file_deck='a',
            sample_rule='random',
            sample_from_distribution=True,
            quadrature=False,
            overwrite=True,
            start_date=datetime(2018, 9, 12),
        )

    for i in range(5):
        assert (output_directory / f'vortex_1_variable_random_{i+1}.json').is_file()
        assert (output_directory / f'vortex_1_variable_random_{i+1}.22').is_file()

        track = VortexTrack.from_file(
            output_directory / f'vortex_1_variable_random_{i+1}.22',
            advisories=['OFCL'],
            file_deck='a',
        )
        dates = np.array(track.data.track_start_time.unique())
        assert len(dates) == 1
        assert pd.to_datetime(dates.item()) == datetime(2018, 9, 12)
