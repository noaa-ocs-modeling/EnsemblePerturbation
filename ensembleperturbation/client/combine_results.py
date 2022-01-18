from argparse import ArgumentParser
import logging
from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

from pandas import DataFrame

from ensembleperturbation.parsing.adcirc import combine_outputs, ElevationSelection
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing')


def parse_combine_results():
    cwd = Path.cwd()

    argument_parser = ArgumentParser()
    argument_parser.add_argument('output', nargs='?', default=cwd, help='output directory')
    argument_parser.add_argument(
        'directory',
        nargs='?',
        default=cwd,
        help='directory containing completed `runs` directory',
    )
    argument_parser.add_argument(
        '--filenames', nargs='*', default=None, help='ADCIRC output files to parse',
    )
    argument_parser.add_argument(
        '--bounds', help='bounding box in format `(minx,miny,maxx,maxy)`'
    )
    argument_parser.add_argument('--max-depth', help='maximum depth value to filter by')
    argument_parser.add_argument(
        '--elevation-selection',
        help='filter elevation nodes based on sea level (one of `wet`, `inundated`, `dry`)',
    )
    argument_parser.add_argument(
        '--parallel', action='store_true', help='load concurrently with Dask'
    )
    argument_parser.add_argument(
        '--verbose', action='store_true', help='log more verbose messages'
    )
    arguments = argument_parser.parse_args()

    return {
        'output': arguments.output,
        'directory': arguments.directory,
        'filenames': arguments.filenames,
        'max_depth': float(arguments.max_depth) if arguments.max_depth is not None else None,
        'bounds': arguments.bounds,
        'elevation_selection': arguments.elevation_selection,
        'parallel': arguments.parallel,
        'verbose': arguments.verbose,
    }


def combine_results(
    output: PathLike,
    directory: PathLike = None,
    filenames: List[str] = None,
    bounds: Tuple[float, float, float, float] = None,
    max_depth: float = None,
    elevation_selection: ElevationSelection = None,
    parallel: bool = False,
    verbose: bool = False,
) -> Dict[str, DataFrame]:
    if verbose:
        get_logger(LOGGER.name, console_level=logging.DEBUG)

    parsed_data = combine_outputs(
        directory,
        file_data_variables=filenames,
        bounds=bounds,
        maximum_depth=max_depth,
        elevation_selection=elevation_selection,
        output_directory=output,
        parallel=parallel,
    )

    return parsed_data


def main():
    combine_results(**parse_combine_results())


if __name__ == '__main__':
    main()
