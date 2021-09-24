from pathlib import Path

import chaospy
from matplotlib import pyplot
import numpy
import xarray

from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.uncertainty_quantification.ensemble_array import read_combined_hdf

if __name__ == '__main__':
    plot = False

    input_filename = r'C:\Data\COASTAL_Act\runs\run_20210902_florence_multivariate_besttrack_250msubset_40members_nws8_william\run_20210902_florence_multivariate_besttrack_250msubset_40members_nws8_william.h5'

    if not isinstance(input_filename, Path):
        input_filename = Path(input_filename)

    netcdf_filename = input_filename.parent / (input_filename.stem + '.nc')

    if not netcdf_filename.exists():
        raise ValueError(f'no NetCDF4 found at "{netcdf_filename}"')

    netcdf_dataset = xarray.open_dataset(netcdf_filename)

    variables = {
        variable_class.name: variable_class()
        for variable_class in VortexPerturbedVariable.__subclasses__()
    }

    dataframes = read_combined_hdf(filename=input_filename)
    ensemble_perturbations = dataframes['vortex_perturbation_parameters']
    mesh_zeta_max = dataframes['zeta_max']

    distribution = chaospy.J(
        *(
            variables[variable_name].chaospy_distribution()
            for variable_name in ensemble_perturbations.columns
        )
    )

    # get samples of the model at certain times and nodes
    times = netcdf_dataset['time']
    grid_nodes = netcdf_dataset['node'][::5]
    samples = netcdf_dataset.loc[{'time': times, 'node': grid_nodes}]['zeta']

    # expand polynomials with polynomial chaos
    polynomials = chaospy.generate_expansion(
        order=3, dist=distribution, rule='three_terms_recurrence',
    )

    # create surrogate models for selected nodes
    surrogate_models = {
        index: chaospy.fit_quadrature(
            orth=polynomials,
            nodes=ensemble_perturbations.iloc[:, :-1].values,
            weights=ensemble_perturbations.iloc[:, -1].values,
            solves=grid_node,
        )
        for index, grid_node in enumerate(samples.T)
        if numpy.any(~numpy.isnan(grid_node))
    }

    for surrogate_model in surrogate_models:
        mean = chaospy.E(poly=surrogate_model, dist=distribution)
        deviation = chaospy.Std(poly=surrogate_model, dist=distribution)

        if plot:
            pyplot.plot(times, mean)
            pyplot.fill_between(
                times, mean - deviation, mean + deviation, alpha=0.5,
            )
            pyplot.show()

        c_1 = [1, 0.9, 0.07]

        y_1 = distribution.fwd(c_1)
        assert distribution.inv(y_1) == c_1

    print('done')
