import unittest

import numpy
from pyproj import CRS, Transformer

from ensemble_perturbation.configuration.seabed.ngdc import NGDCSeabedDescriptions


class TestSeabedDescriptions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crs = CRS.from_epsg(32618)

        self.surveys = NGDCSeabedDescriptions.all_surveys()[:5]

        bounds = numpy.array([[-77, 39], [-75, 40]])
        transformer = Transformer.from_crs(CRS.from_epsg(4326), self.crs)
        self.bounds = numpy.ravel(
            numpy.stack(transformer.transform(bounds[:, 0], bounds[:, 1]), axis=1)
        )

    def test_seabed_descriptions(self):
        seabed = NGDCSeabedDescriptions(bounds=self.bounds, surveys=self.surveys, crs=self.crs)

        self.assertGreater(seabed.data.shape[0], 0)
        self.assertEqual(14, seabed.data.shape[1])
        self.assertGreater(len(seabed.descriptions), 0)
        self.assertTrue(any(seabed.data['Survey'].isin(['492'])))


if __name__ == '__main__':
    unittest.main()
