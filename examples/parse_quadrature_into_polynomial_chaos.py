from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import geopandas
from matplotlib import pyplot
import xarray

from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable
from ensembleperturbation.uncertainty_quantification.ensemble_array import read_combined_hdf

if __name__ == '__main__':
    plot = True

    input_filename = r'C:\Data\COASTAL_Act\runs\run_20210928_florence_besttrack_250msubset_quadrature_manual\run_20210928_florence_besttrack_250msubset_quadrature_manual.h5'
    if not isinstance(input_filename, Path):
        input_filename = Path(input_filename)

    netcdf_filename = input_filename.parent / 'fort.63.nc'
    if not netcdf_filename.exists():
        raise ValueError(f'no NetCDF4 found at "{netcdf_filename}"')

    netcdf_dataset = xarray.open_dataset(netcdf_filename)

    variables = {
        variable_class.name: variable_class()
        for variable_class in VortexPerturbedVariable.__subclasses__()
    }

    dataframes = read_combined_hdf(filename=input_filename)
    ensemble_perturbations = dataframes['vortex_perturbation_parameters']

    distribution = chaospy.J(
        *(
            variables[variable_name].chaospy_distribution()
            for variable_name in ensemble_perturbations.columns[:-1]
        )
    )

    # sample times and nodes
    # TODO: sample based on sentivity / eigenvalues
    sample_times = netcdf_dataset['time']
    sample_nodes = netcdf_dataset['node']
    samples = netcdf_dataset['zeta'].loc[{'time': sample_times, 'node': sample_nodes}]

    if plot:
        storm_name = 'florence2018'

        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)

        countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        countries.plot(color='lightgrey', ax=axis)

        axis.scatter(samples['x'], samples['y'], c=samples.max('time').std('run'))
        figure.suptitle(
            f'standard deviation of max elevation across {len(samples["run"])} run(s) and {len(samples["time"])} time(s)'
        )

        storm = BestTrackForcing(storm_name)
        storm.data.plot(x='longitude', y='latitude', label=storm_name, ax=axis)

        pyplot.show()

    # expand polynomials with polynomial chaos
    polynomials = chaospy.generate_expansion(
        order=3, dist=distribution, rule='three_terms_recurrence',
    )

    # create surrogate models for selected nodes
    surrogate_model = chaospy.fit_quadrature(
        orth=polynomials,
        nodes=ensemble_perturbations.iloc[:, :-1].T.values,
        weights=ensemble_perturbations.iloc[:, -1].values,
        solves=samples.T,
    )

    # surrogate_models = {}
    # with ProcessPoolExecutorStackTraced() as process_pool:
    #     partial_fit_quadrature = partial(
    #         chaospy.fit_quadrature,
    #         orth=polynomials,
    #         nodes=ensemble_perturbations.iloc[:, :-1].values,
    #         weights=ensemble_perturbations.iloc[:, -1].values,
    #     )
    #     futures = {
    #         process_pool.submit(partial_fit_quadrature, solves=grid_node): index
    #         for index, grid_node in enumerate(samples.T)
    #     }
    #
    #     for completed_future in concurrent.futures.as_completed(futures):
    #         surrogate_models[futures[completed_future]] = completed_future.result()

    # for surrogate_model in surrogate_models:
    mean = chaospy.E(poly=surrogate_model, dist=distribution)
    deviation = chaospy.Std(poly=surrogate_model, dist=distribution)

    if plot:
        pyplot.plot(sample_times, mean)
        pyplot.fill_between(
            sample_times, mean - deviation, mean + deviation, alpha=0.5,
        )
        pyplot.show()

    c_1 = [1, 0.9, 0.07]

    y_1 = distribution.fwd(c_1)
    assert distribution.inv(y_1) == c_1

    print('done')
