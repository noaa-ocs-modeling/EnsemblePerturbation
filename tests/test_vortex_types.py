from dateutil.parser import parse as parse_date
from ensembleperturbation.tropicalcyclone.atcf import VortexForcing

from tests import check_reference_directory, DATA_DIRECTORY


def test_vortex_types():
    output_directory = DATA_DIRECTORY / 'output' / 'test_vortex_types'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_vortex_types'

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    start_date = parse_date('2018-09-11 06:00')
    end_date_dict = {'a': None,
                     'b': parse_date('2018-09-18 06:00')
                     }
    record_types_dict = {'a': ['OFCL','HWRF','HMON','CARQ'],
                         'b': ['BEST']
                         }

    for file_deck in ['a','b']:
        record_types_list = record_types_dict[file_deck]
        for record_type in record_types_list:
            cyclone = VortexForcing('al062018',
                start_date=start_date,end_date=end_date_dict[file_deck],
                file_deck=file_deck,requested_record_type=record_type
            )

            filename = file_deck + '-deck_' + record_type + '.txt'
            output_file = output_directory / filename 
            cyclone.write(output_file,overwrite=True)

    check_reference_directory(output_directory, reference_directory)
