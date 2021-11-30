from importlib import metadata as importlib_metadata
import os
from os import PathLike
from pathlib import Path
import re
from typing import Dict, List

import xarray

DATA_DIRECTORY = Path(__file__).parent / 'data'


def package_is_installed(name: str, version: str = None) -> bool:
    installed_distributions = importlib_metadata.distributions()
    for distribution in installed_distributions:
        installed_name = distribution.metadata['Name']
        installed_version = distribution.version
        if installed_name is not None and installed_name.lower() == name.lower():
            if version is None or (
                installed_version is not None and version.lower() == installed_version.lower()
            ):
                return True
    else:
        return False


def check_reference_directory(
    test_directory: PathLike,
    reference_directory: PathLike,
    skip_lines: Dict[str, List[int]] = None,
):
    if not isinstance(test_directory, Path):
        test_directory = Path(test_directory)
    if not isinstance(reference_directory, Path):
        reference_directory = Path(reference_directory)
    if skip_lines is None:
        skip_lines = {}

    for reference_filename in reference_directory.iterdir():
        if reference_filename.is_dir():
            check_reference_directory(
                test_directory / reference_filename.name, reference_filename, skip_lines
            )
        else:
            test_filename = test_directory / reference_filename.name

            if reference_filename.suffix in ['.h5', '.nc']:
                if package_is_installed('xarray'):
                    try:
                        reference_dataset = xarray.open_dataset(reference_filename)
                        test_dataset = xarray.open_dataset(test_filename)
                        return test_dataset == reference_dataset
                    except:
                        pass

                reference_filesize = Path(reference_filename).stat().st_size
                test_filesize = Path(test_filename).stat().st_size

                diff = test_filesize - reference_filesize
                message = f'"{test_filesize}" != "{reference_filesize}"\n{diff}'

                assert reference_filesize == test_filesize, message
                continue

            with open(test_filename) as test_file, open(reference_filename) as reference_file:
                test_lines = list(test_file.readlines())
                reference_lines = list(reference_file.readlines())

                lines_to_skip = set()
                for file_mask, line_indices in skip_lines.items():
                    if (
                        file_mask in str(test_filename)
                        or re.match(file_mask, str(test_filename))
                        and len(test_lines) > 0
                    ):
                        try:
                            lines_to_skip.update(
                                line_index % len(test_lines) for line_index in line_indices
                            )
                        except ZeroDivisionError:
                            continue

                for line_index in sorted(lines_to_skip, reverse=True):
                    del test_lines[line_index], reference_lines[line_index]

                cwd = Path.cwd()
                assert '\n'.join(test_lines) == '\n'.join(
                    reference_lines
                ), f'"{os.path.relpath(test_filename, cwd)}" != "{os.path.relpath(reference_filename, cwd)}"'
