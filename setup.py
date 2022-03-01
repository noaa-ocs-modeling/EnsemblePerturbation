import os
from pathlib import Path
import subprocess
import sys

from setuptools import config, find_packages, setup

DEPENDENCIES = {
    'adcircpy>=1.1.0': ['gdal', 'fiona'],
    'appdirs': [],
    'cartopy': ['cython', 'numpy', 'proj'],
    'chaospy': [],
    'cmocean': [],
    'bs4': [],
    'coupledmodeldriver>=1.4.11': [],
    'dask': [],
    'fiona': ['gdal'],
    'geopandas': [],
    'matplotlib': [],
    'netcdf4': [],
    'numpy': [],
    'pandas': [],
    'pint': [],
    'pint-pandas': ['pint'],
    'pyproj>=2.6': [],
    'tables': [],
    'typepigeon': [],
    'python-dateutil': [],
    'requests': [],
    'shapely': [],
    'scikit-learn': [],
    'stormevents': [],
}

if (Path(sys.prefix) / 'conda-meta').exists() or os.name == 'nt':
    try:
        import gartersnake

        MISSING_DEPENDENCIES = gartersnake.missing_requirements(DEPENDENCIES)

        if len(MISSING_DEPENDENCIES) > 0:
            print(f'{len(MISSING_DEPENDENCIES)} (out of {len(DEPENDENCIES)}) dependencies are missing')

        if len(MISSING_DEPENDENCIES) > 0 and gartersnake.is_conda():
            gartersnake.install_conda_requirements(MISSING_DEPENDENCIES)
            MISSING_DEPENDENCIES = gartersnake.missing_requirements(DEPENDENCIES)

        if len(MISSING_DEPENDENCIES) > 0 and gartersnake.is_windows():
            gartersnake.install_windows_requirements(MISSING_DEPENDENCIES)
            MISSING_DEPENDENCIES = gartersnake.missing_requirements(DEPENDENCIES)
    except:
        pass

try:
    try:
        import dunamai
    except ImportError:
        subprocess.run(
            f'{sys.executable} -m pip install dunamai',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    from dunamai import Version

    __version__ = Version.from_any_vcs().serialize()
except (ModuleNotFoundError, RuntimeError) as error:
    print(error)
    __version__ = '0.0.0'

print(f'using version {__version__}')

metadata = config.read_configuration('setup.cfg')['metadata']

setup(
    **metadata,
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    setup_requires=['dunamai', 'setuptools>=41.2'],
    install_requires=list(DEPENDENCIES),
    extras_require={
        'testing': ['pytest', 'pytest-cov', 'pytest-xdist', 'wget'],
        'development': ['flake8', 'isort', 'oitnb'],
        'documentation': [
            'm2r2',
            'sphinx',
            'sphinx-rtd-theme',
            'sphinxcontrib-programoutput',
            'sphinxcontrib-bibtex',
        ],
    },
    entry_points={
        'console_scripts': [
            'make_storm_ensemble=ensembleperturbation.client.make_storm_ensemble:main',
            'perturb_tracks=ensembleperturbation.client.perturb_tracks:main',
            'combine_results=ensembleperturbation.client.combine_results:main',
            'plot_results=ensembleperturbation.client.plot_results:main',
        ],
    },
)
