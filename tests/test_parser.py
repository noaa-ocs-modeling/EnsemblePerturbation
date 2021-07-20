import re

from ensembleperturbation.parsing.adcirc import parse_adcirc_output, parse_adcirc_outputs
from tests import check_reference_directory, DATA_DIRECTORY


def test_parse_adcirc_output():
    input_directory = DATA_DIRECTORY / 'input' / 'test_parse_adcirc_output'
    output_filenames = [
        filename.name
        for filename in input_directory.iterdir()
        if re.match('\.6(0-9)?\.nc', str(filename))
    ]

    output_data = parse_adcirc_output(input_directory)
    for data_variable in output_filenames:
        assert data_variable in output_data


def test_assemble_adcirc_output():
    input_directory = DATA_DIRECTORY / 'input'
    output_directory = DATA_DIRECTORY / 'output' / 'test_assemble_adcirc_output'
    reference_directory = DATA_DIRECTORY / 'reference' / 'test_assemble_adcirc_output'
    output_filetypes = {
        'maxele.63.nc': 'zeta_max',
        'maxvel.63.nc': 'vel_max',
    }
    max_depth = 5.0  # maximum depth [m] for subsetting the mesh outputs
    write_mode = 'w'  # make/overwrite to a new HDF5 file

    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    output_data = parse_adcirc_outputs(
        directory=input_directory, file_data_variables=output_filetypes.keys(),
    )

    for perturbation in output_data:
        for variable in output_data[perturbation]:
            ds = output_data[perturbation][variable]
            depth = ds['depth']
            subset = depth < max_depth
            ds_subset = ds[['x', 'y', 'depth', output_filetypes[variable]]][subset]
            ds_subset.to_hdf(
                output_directory / f'{output_filetypes[variable]}.h5',
                perturbation,
                mode=write_mode,
                format='table',
                data_columns=True,
            )
        write_mode = 'a'  # switch to append for subsequent perturbations

    check_reference_directory(output_directory, reference_directory)
