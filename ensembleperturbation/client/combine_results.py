from argparse import ArgumentParser
import logging
from pathlib import Path

from pandas import DataFrame

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.utilities import get_logger

LOGGER = get_logger('parsing')


def main() -> {str: DataFrame}:
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

    if arguments.verbose:
        get_logger(LOGGER.name, console_level=logging.DEBUG)

    variable_dataframe = combine_outputs(
        arguments.directory,
        maximum_depth=arguments.max_depth,
        bounds=arguments.bounds,
        output_filename=arguments.output,
    )

    return variable_dataframe


if __name__ == '__main__':
    main()
