import os
from pathlib import Path
import tarfile

import requests


def download_test_configuration(directory: str):
    """
    fetch shinnecock inlet test data
    :param directory: local directory
    """

    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        os.makedirs(directory, exist_ok=True)

    url = "https://www.dropbox.com/s/1wk91r67cacf132/" \
          "NetCDF_shinnecock_inlet.tar.bz2?dl=1"
    remote_file = requests.get(url, stream=True)
    temporary_filename = directory / 'temp.tar.gz'
    with open(temporary_filename, 'b+w') as local_file:
        local_file.write(remote_file.raw.read())
    with tarfile.open(temporary_filename, "r:bz2") as local_file:
        local_file.extractall(directory)
    os.remove(temporary_filename)
