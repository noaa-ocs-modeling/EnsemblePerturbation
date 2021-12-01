Perturbation
============

storm vortex tracks (ATCF)
--------------------------

This module provides functionality to

1. extract an ATCF best track dataset (``fort.22``)
2. randomly perturb different parameters (intensity, size, track coordinates) to generate an ensemble of tracks
3. write out each ensemble member to the ATCF tropical cyclone vortex file (``fort.22``)

authors:

- William Pringle, Argonne National Laboratory, Mar-May 2021
- Zach Burnett, NOS/NOAA
- Saeed Moghimi, NOS/NOAA

perturbed variables
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.perturbation.atcf.MaximumSustainedWindSpeed
.. autoclass:: ensembleperturbation.perturbation.atcf.RadiusOfMaximumWinds
.. autoclass:: ensembleperturbation.perturbation.atcf.CrossTrack
.. autoclass:: ensembleperturbation.perturbation.atcf.AlongTrack

non-perturbed variables
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.perturbation.atcf.CentralPressure
.. autoclass:: ensembleperturbation.perturbation.atcf.BackgroundPressure

perturber class
^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.perturbation.atcf.VortexPerturber
