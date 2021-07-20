#!/usr/bin/env python
import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys

from setuptools import config, find_packages, setup

BUILT_PACKAGES = {'fiona': ['gdal'], 'numpy': [], 'pyproj': ['gdal'], 'shapely': ['gdal']}
is_conda = (Path(sys.prefix) / 'conda-meta').exists()

if is_conda:
    conda_packages = []
    for conda_package in BUILT_PACKAGES:
        try:
            importlib.import_module(conda_package)
        except:
            conda_packages.append(conda_package)
    if len(conda_packages) > 0:
        subprocess.check_call(['conda', 'install', '-y', *conda_packages])

if os.name == 'nt':
    for required_package, pipwin_dependencies in BUILT_PACKAGES.items():
        try:
            importlib.import_module(required_package)
        except:
            try:
                import pipwin
            except:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pipwin'])

            failed_pipwin_packages = []
            for pipwin_package in pipwin_dependencies + [required_package]:
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pipwin', 'install', pipwin_package.lower()]
                    )
                except subprocess.CalledProcessError:
                    failed_pipwin_packages.append(pipwin_package)

            if len(failed_pipwin_packages) > 0:
                raise RuntimeError(
                    f'failed to download or install non-conda Windows build(s) of {" and ".join(failed_pipwin_packages)}; you can either\n'
                    '1) install within an Anaconda environment, or\n'
                    f'2) `pip install <file>.whl`, with `<file>.whl` downloaded from {" and ".join("https://www.lfd.uci.edu/~gohlke/pythonlibs/#" + value.lower() for value in failed_pipwin_packages)} for your Python version'
                )

try:
    try:
        from dunamai import Version
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dunamai'])
        from dunamai import Version

    version = Version.from_any_vcs().serialize()
except RuntimeError as error:
    logging.exception(error)
    version = '0.0.0'

logging.info(f'using version {version}')

metadata = config.read_configuration('setup.cfg')['metadata']

setup(
    name=metadata['name'],
    version=version,
    author=metadata['author'],
    author_email=metadata['author_email'],
    description=metadata['description'],
    long_description=metadata['long_description'],
    long_description_content_type='text/markdown',
    url=metadata['url'],
    packages=find_packages(),
    python_requires='>=3.6',
    setup_requires=['dunamai', 'setuptools>=41.2'],
    install_requires=[
        'adcircpy>=1.0.39',
        'appdirs',
        'bs4',
        'click',
        'coupledmodeldriver>=1.4.3',
        'fiona',
        'geopandas',
        'matplotlib',
        'nemspy>=0.6.16',
        'netcdf4',
        'numpy',
        'pandas',
        'pint',
        'pint-pandas',
        'pyproj>=2.6',
        'python-dateutil',
        'requests',
        'shapely',
    ],
    extras_require={
        'testing': ['pytest', 'pytest-cov', 'pytest-mock', 'pytest-xdist', 'tables', 'wget'],
        'development': ['flake8', 'isort', 'oitnb'],
    },
    entry_points={'console_scripts': ['make_storm_ensemble=client.make_storm_ensemble:main'],},
)
