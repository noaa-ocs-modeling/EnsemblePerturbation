from collections.abc import Mapping
import logging
import os
from pathlib import Path
import re
import subprocess
import sys

from setuptools import config, find_packages, setup

DEPENDENCIES = {
    'adcircpy>=1.0.39': ['gdal', 'fiona'],
    'appdirs': [],
    'bs4': [],
    'click': [],
    'coupledmodeldriver>=1.4.6': [],
    'fiona': ['gdal'],
    'geopandas': [],
    'matplotlib': [],
    'nemspy>=0.6.16': [],
    'netcdf4': [],
    'numpy': [],
    'pandas': [],
    'pint': [],
    'pint-pandas': [],
    'pyproj>=2.6': [],
    'python-dateutil': [],
    'requests': [],
    'shapely': [],
}


def installed_packages() -> [str]:
    return [
        re.split('#egg=', re.split('==| @ ', package.decode())[0])[-1].lower()
        for package in subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze']
        ).splitlines()
    ]


def missing_packages(required_packages: {str: [str]}) -> {str: [str]}:
    if isinstance(required_packages, Mapping):
        missing_dependencies = missing_packages(list(required_packages))
        output = {}
        for dependency, subdependencies in required_packages.items():
            missing_subdependencies = missing_packages(subdependencies)
            if dependency in missing_dependencies or len(missing_subdependencies) > 0:
                output[dependency] = missing_subdependencies
        return output
    else:
        return [
            required_package
            for required_package in required_packages
            if re.split('<|<=|==|>=|>', required_package)[0].lower()
            not in installed_packages()
        ]


MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

if (Path(sys.prefix) / 'conda-meta').exists() and len(MISSING_DEPENDENCIES) > 0:
    try:
        subprocess.check_call(['conda', 'install', '-y', list(MISSING_DEPENDENCIES)])
    except:
        for dependency in list(MISSING_DEPENDENCIES):
            try:
                subprocess.check_call(['conda', 'install', '-y', dependency])
            except:
                continue

    MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

if os.name == 'nt' and len(MISSING_DEPENDENCIES) > 0:
    if 'pipwin' not in installed_packages():
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pipwin'])
    subprocess.check_call([sys.executable, '-m', 'pipwin', 'refresh'])

    for dependency, subdependencies in MISSING_DEPENDENCIES.items():
        failed_pipwin_packages = []
        for _ in range(1 + len(subdependencies)):
            for package_name in subdependencies + [dependency]:
                if dependency in missing_packages(
                    DEPENDENCIES
                ) or package_name in missing_packages(subdependencies):
                    try:
                        subprocess.check_call(
                            [sys.executable, '-m', 'pipwin', 'install', package_name.lower()]
                        )
                        if package_name in failed_pipwin_packages:
                            failed_pipwin_packages.remove(package_name)
                    except subprocess.CalledProcessError:
                        try:
                            subprocess.check_call(
                                [sys.executable, '-m', 'pip', 'install', package_name.lower()]
                            )
                        except subprocess.CalledProcessError:
                            failed_pipwin_packages.append(package_name)

            # since we don't know the dependencies here, repeat this process n number of times
            # (worst case is `O(n)`, where the first package is dependant on all the others)
            if len(failed_pipwin_packages) == 0:
                break

    MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

try:
    if 'dunamai' not in installed_packages():
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
    install_requires=list(DEPENDENCIES),
    extras_require={
        'testing': ['pytest', 'pytest-cov', 'pytest-xdist', 'tables', 'wget'],
        'development': ['flake8', 'isort', 'oitnb'],
    },
    entry_points={
        'console_scripts': [
            'make_storm_ensemble=ensembleperturbation.client.make_storm_ensemble:main',
            'perturb_tracks=ensembleperturbation.client.perturb_tracks:main',
        ],
    },
)
