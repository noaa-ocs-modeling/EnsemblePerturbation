from pathlib import Path

from adcircpy.forcing import BestTrackForcing
import chaospy
import geopandas
from matplotlib import pyplot
import xarray

from ensembleperturbation.parsing.adcirc import combine_outputs
from ensembleperturbation.perturbation.atcf import VortexPerturbedVariable

if __name__ == '__main__':
    plot = False
    input_directory = Path.cwd()

    filenames = ['perturbations.nc', 'fort.63.nc']

    datasets = {}
    existing_filenames = []
    for filename in filenames:
        filename = input_directory / filename
        if filename.exists():
            datasets[filename.name] = xarray.open_dataset(filename)
            existing_filenames.append(filename.name)

    for filename in existing_filenames:
        filenames.remove(filename)

    if len(filenames) > 0:
        datasets.update(
            combine_outputs(
                input_directory,
                file_data_variables=filenames,
                maximum_depth=0,
                only_inundated=True,
                parallel=True,
            )
        )

    perturbations = datasets['perturbations.nc']
    elevations = datasets['fort.63.nc']

    variables = {
        variable_class.name: variable_class()
        for variable_class in VortexPerturbedVariable.__subclasses__()
    }

    distribution = chaospy.J(
        *(
            variables[variable_name].chaospy_distribution()
            for variable_name in perturbations['variable'].values
        )
    )

    # sample times and nodes
    # TODO: sample based on sentivity / eigenvalues
    sample_times = elevations['time'][::10]
    sample_nodes = elevations['node'][::1000]
    samples = elevations['zeta'].loc[{'time': sample_times, 'node': sample_nodes}]
    # samples = elevations['zeta']

    if plot:
        figure = pyplot.figure()
        figure.suptitle(
            f'standard deviation of {len(samples["node"])} max elevation(s) across {len(samples["run"])} run(s) and {len(samples["time"])} time(s)'
        )
        axis = figure.add_subplot(1, 1, 1)

        countries = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        countries.plot(color='lightgrey', ax=axis)

        axis.scatter(samples['x'], samples['y'], c=samples.max('time').std('run'))

        storm_name = 'florence2018'
        storm = BestTrackForcing(storm_name)
        storm.data.plot(x='longitude', y='latitude', label=storm_name, ax=axis)

        pyplot.show()

    # expand polynomials with polynomial chaos
    polynomials = chaospy.generate_expansion(
        order=3, dist=distribution, rule='three_terms_recurrence',
    )

    # create surrogate models for selected nodes
    print(f'fitting surrogate to {samples.shape} samples')
    surrogate_model = chaospy.fit_quadrature(
        orth=polynomials,
        nodes=perturbations['perturbations'].T.values,
        weights=perturbations['weights'].values,
        solves=samples,
    )
    print(surrogate_model)

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
        pyplot.plot(samples['times'], mean)
        pyplot.fill_between(
            samples['times'], mean - deviation, mean + deviation, alpha=0.5,
        )
        pyplot.show()

    c_1 = [1, 0.9, 0.07]

    y_1 = distribution.fwd(c_1)
    assert distribution.inv(y_1) == c_1

    print('done')
