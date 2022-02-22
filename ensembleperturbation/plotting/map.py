import io
from os import PathLike
import pathlib
import zipfile

import appdirs
import geopandas
from matplotlib import pyplot
from matplotlib.axis import Axis
import requests


def download_coastline(overwrite: bool = False) -> pathlib.Path:
    data_directory = pathlib.Path(appdirs.user_data_dir('ne_coastline'))
    if not data_directory.exists():
        data_directory.mkdir(exist_ok=True, parents=True)

    coastline_filename = data_directory / 'ne_110m_coastline.shp'

    if not coastline_filename.exists() or overwrite:
        # download and save if not present
        url = 'http://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip'
        response = requests.get(url, stream=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            for member_filename in zip_file.namelist():
                file_data = zip_file.read(member_filename)
                with open(data_directory / member_filename, 'wb') as output_file:
                    output_file.write(file_data)
        assert coastline_filename.exists(), 'coastline file not downloaded'

    return coastline_filename


def plot_coastline(
    axis: Axis = None, show: bool = False, save_filename: PathLike = None, **kwargs
):
    if axis is None:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

    coastline_filename = download_coastline()
    dataframe = geopandas.read_file(coastline_filename)
    dataframe.plot(ax=axis, **kwargs)

    if save_filename is not None:
        pyplot.savefig(save_filename)

    if show:
        pyplot.show()
