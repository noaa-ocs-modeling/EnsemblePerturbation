import pdb
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from stormevents.nhc.const import RMWFillMethod
from stormevents.nhc.track import VortexTrack
from ensembleperturbation.perturbation.atcf import (
    MaximumSustainedWindSpeed,
    perturb_tracks,
)


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    'storm,year,date1,date2',
    [
        ('irma', 2017, datetime(2017, 9, 10, 0), datetime(2017, 9, 13, 12)),
        ('florence', 2018, datetime(2018, 9, 12, 18), datetime(2018, 9, 19, 0)),
        ('laura', 2020, datetime(2020, 8, 25, 6), datetime(2020, 8, 31, 6)),
        ('idalia', 2023, datetime(2023, 8, 29, 0), datetime(2023, 9, 7, 18)),
    ],
)
def test_perturb_performance(storm, year, date1, date2):

    with tempfile.TemporaryDirectory() as tdir:
        track_path = Path(tdir) / 'track.dat'
        pert_path = Path(tdir) / 'track_perturb_dir'

        if not track_path.exists():
            track = VortexTrack.from_storm_name(
                storm,
                year,
                file_deck='a',
                advisories=['OFCL'],
                rmw_fill=RMWFillMethod.regression_penny_2023,
            )
            track.to_file(track_path)

        perturb_tracks(
            39,
            variables=[MaximumSustainedWindSpeed],
            directory=pert_path,
            storm=track_path,
            sample_from_distribution=True,
            sample_rule='korobov',
            file_deck='a',
            advisories=['OFCL'],
            start_date=date1,
            end_date=date2,
            parallel=False,  # Parallel interfere's with timeout
        )
