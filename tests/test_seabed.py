import unittest

from ensemble_perturbation.inputs.seabed.ngdc import NGDCSeabedDescriptions


class TestSeabedDescriptions(unittest.TestCase):
    def test_seabed_descriptions(self):
        surveys = NGDCSeabedDescriptions.all_surveys()
        seabed = NGDCSeabedDescriptions(surveys=surveys[:5],
                                        bounds=(-76.3, 39.0, -76.35, 39.5))

        assert seabed.data.shape[0] > 0
        assert seabed.data.shape[1] == 15
        assert len(seabed.seabed_descriptions) > 0


if __name__ == '__main__':
    unittest.main()
