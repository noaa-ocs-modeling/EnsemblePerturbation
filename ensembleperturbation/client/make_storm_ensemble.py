from argparse import ArgumentParser

from ensembleperturbation.perturbation.atcf import (
    AlongTrack,
    CrossTrack,
    MaximumSustainedWindSpeed,
    RadiusOfMaximumWinds,
    VortexPerturber,
)


def main():
    ##################################
    # Example calls from command line for 2018 Hurricane Florence:
    # - python3 make_storm_ensemble.py 3 al062018 2018-09-11-06 2018-09-17-06
    # - python3 make_storm_ensemble.py 5 Florence2018 2018-09-11-06
    ##################################

    # Implement argument parsing
    argument_parser = ArgumentParser()
    argument_parser.add_argument('number_of_perturbations', help='number of perturbations')
    argument_parser.add_argument('storm_code', help='storm name/code')
    argument_parser.add_argument('start_date', nargs='?', help='start date')
    argument_parser.add_argument('end_date', nargs='?', help='end date')
    arguments = argument_parser.parse_args()

    # hardcoding variable list for now
    variables = [
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWinds,
        AlongTrack,
        CrossTrack,
    ]

    perturber = VortexPerturber(
        storm=arguments.storm_code,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    perturber.write(
        number_of_perturbations=arguments.number_of_perturbations, variables=variables,
    )


if __name__ == '__main__':
    main()
