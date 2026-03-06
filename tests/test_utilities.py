import numpy
import pytest
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import importlib.resources as resources

from ensembleperturbation.utilities import encode_categorical_values, load_colormap
import ensembleperturbation.assets.colormaps as colormaps_pkg


with resources.as_file(resources.files(colormaps_pkg)) as colormaps_dir:
    colormap_files = list(Path(colormaps_dir).glob('*.npy'))

assert len(colormap_files) > 0, 'No colormaps found in assets/colormaps'


def test_encode_categorical_values():
    values_1 = ['apple', 'banana', 'apple', 'orange']
    values_2 = [-5, 20, -1, 0, 0]
    values_3 = numpy.array([[-100, 100], [0, 0], [100, 0],])

    encoded_1 = encode_categorical_values(values_1)
    encoded_2 = encode_categorical_values(values_2)
    encoded_3 = encode_categorical_values(values_3)

    assert numpy.all(encoded_1 == [0, 1, 0, 2])
    assert numpy.all(encoded_2 == [0, 3, 1, 2, 2])
    assert numpy.all(encoded_3 == [[0, 2], [1, 1], [2, 1]])


@pytest.mark.parametrize('cmap_file', colormap_files)
def test_load_colormap(cmap_file):
    """
    Test that a single .npy colormap loads as LinearSegmentedColormap.
    """
    cmap = load_colormap(cmap_file)

    assert isinstance(
        cmap, LinearSegmentedColormap
    ), f'{cmap_file.name} did not load correctly'

    assert cmap.name == cmap_file.stem, f'{cmap_file.name} name mismatch'

    sample = cmap([0.0, 0.5, 1.0])[:, :3]
    assert sample.shape[1] == 3, f'{cmap_file.name} sample should have 3 channels'
    assert numpy.all(
        (sample >= 0) & (sample <= 1)
    ), f'{cmap_file.name} has values out of range'
