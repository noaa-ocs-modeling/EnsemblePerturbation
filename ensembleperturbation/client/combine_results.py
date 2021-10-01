from argparse import ArgumentParser
import logging
from os import PathLike
from pathlib import Path

from pandas import DataFrame

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing')


def parse_combine_results():
    cwd = Path.cwd()

    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        'output', nargs='?', default=cwd / (cwd.name + '.h5'), help='output filename (`*.h5`)'
    )
    argument_parser.add_argument(
        'directory',
        nargs='?',
        default=cwd,
        help='directory containing completed `runs` directory',
    )
    argument_parser.add_argument(
        '--filenames', nargs='*', default=None, help='ADCIRC output files to parse',
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
        'filenames': arguments.filenames,
        'max_depth': float(arguments.max_depth) if arguments.max_depth is not None else None,
        'bounds': arguments.bounds,
        'verbose': arguments.verbose,
    }


def combine_results(
    output: PathLike,
    directory: PathLike = None,
    filenames: [str] = None,
    max_depth: float = None,
    bounds: (float, float, float, float) = None,
    verbose: bool = False,
) -> {str: DataFrame}:
    if verbose:
        get_logger(LOGGER.name, console_level=logging.DEBUG)

    variable_dataframes = combine_outputs(
        directory,
        file_data_variables=filenames,
        maximum_depth=max_depth,
        bounds=bounds,
        output_filename=output,
    )

    return variable_dataframes


def main():
    combine_results(**parse_combine_results())


if __name__ == '__main__':
    main()
