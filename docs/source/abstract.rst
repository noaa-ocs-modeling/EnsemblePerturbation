Abstract
========

The Hurricane Supplemental project is tasked with developing an ensemble of current and near-term future hurricane path, intensity, and size that efficiently represents the uncertainty of wind forcing that generates storm surge and combines with tides to produce coastal flooding.
The resultant simulated coastal flooding estimates from the ensemble are then analyzed to produce a probabilistic form that is useful to emergency services.
After an ensemble is perturbed, generated, and run, uncertainty can be quantified by fitting a surrogate (approximate) model, and querying the surrogate for statistical characteristics using wrappers for various UQ libraries that are built into EnsemblePerturbation.
The surrogate can be a polynomial approximation (or collection of polynomial approximations), a neural network, or a simplified model that exhibits the same behaviour as the overall coupled system without the computational overhead and complexity.
After the surrogate model is created, it can be queried for sensitivities in individual perturbed variables, expected performance as compared with the coupled system, and percentiles of inundation along selected mesh nodes in the model study area.
