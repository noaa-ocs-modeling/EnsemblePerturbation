Output Parsing
==============

ADCIRC outputs
--------------

ADCIRC generates `several output data files <https://adcirc.org/home/documentation/users-manual-v52/output-file-descriptions>`_, depending on configuration.
These data files contain oceanic variables along the model mesh / recording stations, and can be parsed into dedicated Python objects using the ``parse_adcirc_outputs`` function.

.. autofunction:: ensembleperturbation.parsing.adcirc.parse_adcirc_outputs

``fort.61`` - sea-surface elevation at stations (specific stations over time series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.ElevationStationOutput

``fort.62`` - sea-surface horizontal velocity at stations (specific stations over time series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.VelocityStationOutput

``fort.63`` - sea-surface elevation (2D field over time series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.ElevationTimeSeriesOutput

``fort.64`` - sea-surface horizontal velocity (2D field over time series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.VelocityTimeSeriesOutput


``maxele.63`` - maximum sea-surface elevation (2D field with single time per-node)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.MaximumElevationOutput

``maxvel.63`` - maximum sea-surface horizontal velocity (2D field with single time per-node)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.MaximumVelocityOutput

``minpr.63`` - minimum sea-surface atmospheric pressure (2D field with single time per-node)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.MinimumSurfacePressureOutput

``maxwvel.63`` - maximum vertical velocity (2D field with single time per-node)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.MaximumSurfaceWindOutput

``maxrs.63`` - maximum sea-surface wave stress (2D field with single time per-node)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.MaximumSurfaceRadiationStressOutput

``fort.67`` / ``fort.68`` - hot start continuation storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.HotStartOutput
.. autoclass:: ensembleperturbation.parsing.adcirc.HotStartOutput2

``fort.73`` - sea-surface atmospheric pressure (2D field over time series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.SurfacePressureTimeSeriesOutput

``fort.74`` - sea-surface atmospheric wind stress + wind velocity (2D field over time series)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.SurfaceWindTimeSeriesOutput

abstract classes
^^^^^^^^^^^^^^^^

.. autoclass:: ensembleperturbation.parsing.adcirc.AdcircOutput
.. autoclass:: ensembleperturbation.parsing.adcirc.TimeSeriesOutput
.. autoclass:: ensembleperturbation.parsing.adcirc.StationTimeSeriesOutput
.. autoclass:: ensembleperturbation.parsing.adcirc.FieldOutput
.. autoclass:: ensembleperturbation.parsing.adcirc.FieldTimeSeriesOutput
