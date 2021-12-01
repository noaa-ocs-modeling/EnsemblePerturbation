Introduction
============

The Hurricane Supplemental and COASTAL Act projects are tasked with quantifying the uncertainty of coastal flooding from storm surge during named storm events (hurricanes and tropical cyclones).

This is accomplished by developing an ensemble of storm scenarios representing past, current, and near-future hurricane variables (namely path, intensity, and size) to efficiently determine the uncertainty of wind forcing that, along with tides, contributes to storm surge and coastal flooding.
The resulting simulation of coastal flooding estimates from the ensemble are then analyzed to produce a probabilistic form that is useful to emergency services.

After an ensemble is perturbed, generated, and run, uncertainty can be quantified by fitting a surrogate (approximate) model and querying it for statistical characteristics.
The surrogate can be a polynomial approximation (or collection of polynomial approximations), a neural network, or any simplified model that exhibits the same behaviour as the overall coupled system without the computational overhead and complexity.
After the surrogate model is created, it can be queried for sensitivities in individual perturbed variables, expected performance as compared with the coupled system, and percentiles of inundation along selected mesh nodes in the model study area.

``ensembleperturbation`` accomplishes this by utilizing various uncertainty quantification (UQ) libraries, including ``UQtk`` and ``chaospy``.
