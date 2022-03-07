from collections.abc import Mapping
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # for Python<3.8
    subprocess.run(
        f'{sys.executable} -m pip install importlib_metadata',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import importlib_metadata

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


def installed_packages() -> List[str]:
    installed_distributions = importlib_metadata.distributions()
    return [
        distribution.metadata['Name'].lower()
        for distribution in installed_distributions
        if distribution.metadata['Name'] is not None
    ]


def missing_packages(required_packages: Dict[str, List[str]]) -> Dict[str, List[str]]:
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


try:
    if 'dunamai' not in installed_packages():
        subprocess.run(
            f'{sys.executable} -m pip install dunamai',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    from dunamai import Version

    version = Version.from_any_vcs().serialize()
except (ModuleNotFoundError, RuntimeError) as error:
    print(error)
    version = '0.0.0'

print(f'using version {version}')

MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

if len(MISSING_DEPENDENCIES) > 0:
    print(
        f'found {len(MISSING_DEPENDENCIES)} (out of {len(DEPENDENCIES)}) missing dependencies'
    )

if (Path(sys.prefix) / 'conda-meta').exists() and len(MISSING_DEPENDENCIES) > 0:
    print(f'found conda environment at {sys.prefix}')

    if 'tables' in MISSING_DEPENDENCIES:
        MISSING_DEPENDENCIES['pytables'] = MISSING_DEPENDENCIES['tables']
        del MISSING_DEPENDENCIES['tables']

    conda_packages = []
    try:
        subprocess.check_output(
            f'conda install -y {" ".join(MISSING_DEPENDENCIES)}',
            shell=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as error:
        output = error.output.decode()
        package_not_found_start = 'PackagesNotFoundError: The following packages are not available from current channels:\n\n'
        package_not_found_stop = '\n\nCurrent channels:'
        if package_not_found_start in output:
            non_conda_packages = [
                package.replace('-', '').strip()
                for package in output[
                    output.index(package_not_found_start) : output.index(
                        package_not_found_stop
                    )
                ].splitlines()[2:]
            ]
            conda_packages = [
                package
                for package in MISSING_DEPENDENCIES
                if package not in non_conda_packages
            ]

            print(
                f'found {len(conda_packages)} conda packages (out of {len(MISSING_DEPENDENCIES)})'
            )

    try:
        subprocess.run(
            f'conda install -y {" ".join(conda_packages)}',
            shell=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        for dependency in conda_packages:
            try:
                subprocess.run(
                    f'conda install -y {dependency}', shell=True, stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                continue

    MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

if len(MISSING_DEPENDENCIES) > 0:
    try:
        subprocess.run(
            f'{sys.executable} -m pip install {" ".join(package_name.lower() for package_name in MISSING_DEPENDENCIES)}',
            check=True,
            shell=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        for dependency, subdependencies in MISSING_DEPENDENCIES.items():
            for _ in range(1 + len(subdependencies)):
                MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)
                missing_subdependencies = missing_packages(subdependencies)
                packages = [
                    package_name
                    for package_name in subdependencies + [dependency]
                    if package_name in MISSING_DEPENDENCIES
                    or package_name in missing_subdependencies
                ]
                try:
                    subprocess.run(
                        f'{sys.executable} -m pip install {" ".join(package_name.lower() for package_name in packages)}',
                        check=True,
                        shell=True,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError:
                    pass

    MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

if os.name == 'nt' and len(MISSING_DEPENDENCIES) > 0:
    print(f'attempting to install {len(MISSING_DEPENDENCIES)} packages with `pipwin`')

    if 'pipwin' not in installed_packages():
        subprocess.run(
            f'{sys.executable} -m pip install pipwin',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    subprocess.run(f'{sys.executable} -m pipwin refresh', shell=True)

    for dependency, subdependencies in MISSING_DEPENDENCIES.items():
        failed_pipwin_packages = []
        for _ in range(1 + len(subdependencies)):
            for package_name in subdependencies + [dependency]:
                if dependency in missing_packages(
                    DEPENDENCIES
                ) or package_name in missing_packages(subdependencies):
                    try:
                        subprocess.run(
                            f'{sys.executable} -m pipwin install {package_name.lower()}',
                            check=True,
                            shell=True,
                            stderr=subprocess.DEVNULL,
                        )
                        if package_name in failed_pipwin_packages:
                            failed_pipwin_packages.remove(package_name)
                    except subprocess.CalledProcessError:
                        failed_pipwin_packages.append(package_name)

            # since we don't know the dependencies here, repeat this process n number of times
            # (worst case is `O(n)`, where the first package is dependant on all the others)
            if len(failed_pipwin_packages) == 0:
                break

    MISSING_DEPENDENCIES = missing_packages(DEPENDENCIES)

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
