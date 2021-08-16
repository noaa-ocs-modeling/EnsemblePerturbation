from argparse import ArgumentParser
import logging
from os import PathLike
from pathlib import Path

from pandas import DataFrame

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing')


def parse_combine_results():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('output', help='output filename (`*.h5`)')
    argument_parser.add_argument(
        'directory',
        nargs='?',
        default=Path.cwd(),
        help='directory containing completed `runs` directory',
    )
    argument_parser.add_argument('--max-depth', help='maximum depth value to filter by')
    argument_parser.add_argument(
        '--bounds', help='bounding box in format `(minx,miny,maxx,maxy)`'
    )
    argument_parser.add_argument(
        '--verbose', action='store_true', help='log more verbose messages'
    )
    arguments = argument_parser.parse_args()

    return {
        'output': arguments.output,
        'directory': arguments.directory,
        'max_depth': arguments.max_depth,
        'bounds': arguments.bounds,
        'verbose': arguments.verbose,
    }


def combine_results(
    output: PathLike,
    directory: PathLike = None,
    max_depth: float = None,
    bounds: (float, float, float, float) = None,
    verbose: bool = False,
) -> {str: DataFrame}:
    if verbose:
        get_logger(LOGGER.name, console_level=logging.DEBUG)

    variable_dataframes = combine_outputs(
        directory, maximum_depth=max_depth, bounds=bounds, output_filename=output,
    )

    return variable_dataframes


def main():
    combine_results(**parse_combine_results())


if __name__ == '__main__':
    main()
