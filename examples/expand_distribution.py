import chaospy
from matplotlib import pyplot
import numpy

from ensembleperturbation.perturbation.atcf import PerturbationType, VortexPerturbedVariable

if __name__ == '__main__':
    distributions = {}
    for perturbed_variable in VortexPerturbedVariable.__subclasses__():
        name = perturbed_variable.name
        perturbation = perturbed_variable.perturbation_type
        if perturbation == PerturbationType.GAUSSIAN:
            distributions[name] = chaospy.Normal(0, 1)
        elif perturbation == PerturbationType.UNIFORM:
            distributions[name] = chaospy.Uniform(-1, 1)

    joint_distribution = chaospy.J(*distributions.values())

    expansion = chaospy.generate_expansion(8, joint_distribution)
    polynomials = expansion[:5].round(8)

    grid = numpy.mgrid[-2:2:100j, -1:1:100j]

    pyplot.contourf(grid[0], grid[1], joint_distribution.pdf(grid), 100)
    pyplot.scatter(*joint_distribution.sample(100, rule='sobol'))

    pyplot.title(', '.join(distributions))

    pyplot.show()

    print('done')
