from argparse import ArgumentParser

from stormevents.nhc.atcf import ATCF_FileDeck
import typepigeon

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
    perturb_tracks,
    RadiusOfMaximumWinds,
)


def main():
    ##################################
    # Example calls from command line for 2018 Hurricane Florence:
    # - python3 make_storm_ensemble.py 3 al062018 2018-09-11-06 2018-09-17-06
    # - python3 make_storm_ensemble.py 5 Florence2018 2018-09-11-06
    ##################################

    # Implement argument parsing
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        '--number-of-perturbations', required=True, type=int, help='number of perturbations'
    )
    argument_parser.add_argument('--storm-code', required=True, help='storm name/code')
    argument_parser.add_argument('--start-date', help='start date')
    argument_parser.add_argument('--end-date', help='end date')
    argument_parser.add_argument(
        '--advisories', nargs='*', help='record type (i.e. `BEST`, `OFCL`)'
    )
    argument_parser.add_argument(
        '--file-deck', help='letter of file deck, one of `a`, `b`, or `f`'
    )
    argument_parser.add_argument('directory', nargs='?', help='output directory')
    arguments = argument_parser.parse_args()

    # hardcoding variable list for now
    variables = [
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWinds,
        AlongTrack,
        CrossTrack,
    ]

    perturb_tracks(
        perturbations=arguments.number_of_perturbations,
        directory=arguments.directory,
        storm=arguments.storm_code,
        variables=variables,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        file_deck=typepigeon.convert_value(arguments.file_deck, ATCF_FileDeck),
        advisories=arguments.advisories,
    )


if __name__ == '__main__':
    main()
