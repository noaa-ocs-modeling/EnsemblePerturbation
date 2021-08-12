from argparse import ArgumentParser
from pathlib import Path

from ensembleperturbation.parsing.adcirc import combine_outputs


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('output-filename', help='output filename (`*.h5`)')
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
    arguments = argument_parser.parse_args()

    dataframe = combine_outputs(
        arguments.directory,
        maximum_depth=arguments.max_depth,
        bounds=arguments.bounds,
        output_filename=arguments.output_filename,
    )

    return dataframe


if __name__ == '__main__':
    main()
