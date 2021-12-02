CLI Commands
============

``make_storm_ensemble``
-----------------------

.. program-output:: make_storm_ensemble -h

.. autofunction:: ensembleperturbation.perturbation.atcf.perturb_tracks

``perturb_tracks``
------------------

``perturb_tracks`` is an extension of ``initialize_adcirc`` `from CoupledModelDriver <https://coupledmodeldriver.readthedocs.io/en/latest/client.html#initialize-adcirc-create-json-configuration-files>`_

.. program-output:: perturb_tracks -h

The extension adds several new arguments: ``--perturbations``, ``--variables``, ``--quadrature``, and ``--serial``

.. code-block:: shell

    perturb_tracks \
        --platform HERA \
        --mesh-directory /scratch2/COASTAL/coastal/save/shared/models/meshes/hsofs/250m/v1.0_fixed \
        --modeled-start-time "2018-09-11 06:00:00" \
        --modeled-duration "07:00:00:00" \
        --modeled-timestep "00:00:02" \
        --nems-interval "01:00:00" \
        --forcings tidal,besttrack \
        --tidal-spinup-duration "07:00:00:00" \
        --tidal-source TPXO \
        --tidal-path /scratch2/COASTAL/coastal/save/shared/models/forcings/tides/h_tpxo9.v1.nc \
        --besttrack-nws 8 \
        --besttrack-storm-id florence2018 \
        --adcirc-executable /scratch2/COASTAL/coastal/save/shared/repositories/CoastalApp_PaHM/ALLBIN_INSTALL/NEMS.x \
        --adcprep-executable /scratch2/COASTAL/coastal/save/shared/repositories/CoastalApp_PaHM/ALLBIN_INSTALL/adcprep \
        --perturbations 256 \
        --quadrature \
        --serial

``combine_results``
-------------------

.. program-output:: combine_results -h

.. autofunction:: ensembleperturbation.client.combine_results.combine_results

``plot_results``
----------------

.. program-output:: plot_results -h
