import sys

import gartersnake
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

MISSING_DEPENDENCIES = gartersnake.missing_requirements(DEPENDENCIES)

if len(MISSING_DEPENDENCIES) > 0:
    print(f'{len(MISSING_DEPENDENCIES)} (out of {len(DEPENDENCIES)}) dependencies are missing')

if len(MISSING_DEPENDENCIES) > 0 and gartersnake.is_conda():
    print(f'found conda environment at {sys.prefix}')
    gartersnake.install_conda_requirements(MISSING_DEPENDENCIES)
    MISSING_DEPENDENCIES = gartersnake.missing_requirements(DEPENDENCIES)

if len(MISSING_DEPENDENCIES) > 0 and gartersnake.is_windows():
    gartersnake.install_windows_requirements(MISSING_DEPENDENCIES)
    MISSING_DEPENDENCIES = gartersnake.missing_requirements(DEPENDENCIES)

__version__ = gartersnake.vcs_version()
print(f'using version {__version__}')

metadata = config.read_configuration('setup.cfg')['metadata']

setup(
    **metadata,
    version=__version__,
    packages=find_packages(),
    python_requires='>=3.6',
    setup_requires=['dunamai', 'gartersnake', 'setuptools>=41.2'],
    install_requires=list(DEPENDENCIES),
    extras_require={
        'testing': ['pytest', 'pytest-cov', 'pytest-xdist', 'wget'],
        'development': ['flake8', 'isort', 'oitnb'],
        'documentation': [
            'dunamai',
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
