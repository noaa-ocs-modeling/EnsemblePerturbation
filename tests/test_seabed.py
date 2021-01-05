import numpy
from pyproj import CRS, Transformer

from ensemble_perturbation.configuration.seabed.ngdc import NGDCSeabedDescriptions

CRS = CRS.from_epsg(32618)
SURVEYS = NGDCSeabedDescriptions.all_surveys()[:5]

BOUNDS = numpy.array([[-77, 39], [-75, 40]])
BOUNDS = numpy.ravel(
    numpy.stack(
        Transformer.from_crs(CRS.from_epsg(4326), CRS).transform(BOUNDS[:, 0], BOUNDS[:, 1]),
        axis=1,
    )
)


def test_seabed_descriptions():
    seabed = NGDCSeabedDescriptions(bounds=BOUNDS, surveys=SURVEYS, crs=CRS)

    assert seabed.data.shape[0] > 0
    assert seabed.data.shape[1] == 14
    assert len(seabed.descriptions) > 0
    assert any(seabed.data['Survey'].isin(['492']))
