# Ensemble Perturbation

[![tests](https://github.com/noaa-ocs-modeling/EnsemblePerturbation/workflows/tests/badge.svg)](https://github.com/noaa-ocs-modeling/EnsemblePerturbation/actions?query=workflow%3Atests)
[![codecov](https://codecov.io/gh/noaa-ocs-modeling/ensembleperturbation/branch/main/graph/badge.svg?token=4DwZePHp18)](https://codecov.io/gh/noaa-ocs-modeling/ensembleperturbation)
[![build](https://github.com/noaa-ocs-modeling/EnsemblePerturbation/workflows/build/badge.svg)](https://github.com/noaa-ocs-modeling/EnsemblePerturbation/actions?query=workflow%3Abuild)
[![version](https://img.shields.io/pypi/v/EnsemblePerturbation)](https://pypi.org/project/EnsemblePerturbation)
[![license](https://img.shields.io/github/license/noaa-ocs-modeling/EnsemblePerturbation)](https://creativecommons.org/share-your-work/public-domain/cc0)
[![style](https://sourceforge.net/p/oitnb/code/ci/default/tree/_doc/_static/oitnb.svg?format=raw)](https://sourceforge.net/p/oitnb/code)

Python library for perturbing coupled model inputs into ensemble runs. Provides perturbation and results comparison.

```bash
pip install ensembleperturbation
```

## Command-line interface

`ensembleperturbation` exposes the following CLI commands:

- `perturb_tracks`
- `make_storm_ensemble`

### Perturb Best Track (`perturb_tracks`)

```
usage: perturb_tracks [-h] --platform PLATFORM --mesh-directory MESH_DIRECTORY --modeled-start-time MODELED_START_TIME
                      --modeled-duration MODELED_DURATION --modeled-timestep MODELED_TIMESTEP
                      [--nems-interval NEMS_INTERVAL] [--modulefile MODULEFILE] [--forcings FORCINGS]
                      [--adcirc-executable ADCIRC_EXECUTABLE] [--adcprep-executable ADCPREP_EXECUTABLE]
                      [--aswip-executable ASWIP_EXECUTABLE] [--adcirc-processors ADCIRC_PROCESSORS]
                      [--job-duration JOB_DURATION] [--output-directory OUTPUT_DIRECTORY] [--skip-existing]
                      [--absolute-paths] [--verbose] [--perturbations PERTURBATIONS] [--variables VARIABLES]

optional arguments:
  -h, --help            show this help message and exit
  --platform PLATFORM   HPC platform for which to configure
  --mesh-directory MESH_DIRECTORY
                        path to input mesh (`fort.13`, `fort.14`)
  --modeled-start-time MODELED_START_TIME
                        start time within the modeled system
  --modeled-duration MODELED_DURATION
                        end time within the modeled system
  --modeled-timestep MODELED_TIMESTEP
                        time interval within the modeled system
  --nems-interval NEMS_INTERVAL
                        main loop interval of NEMS run
  --modulefile MODULEFILE
                        path to module file to `source`
  --forcings FORCINGS   comma-separated list of forcings to configure, from ['tidal', 'atmesh', 'besttrack', 'owi',
                        'ww3data']
  --adcirc-executable ADCIRC_EXECUTABLE
                        filename of compiled `adcirc` or `NEMS.x`
  --adcprep-executable ADCPREP_EXECUTABLE
                        filename of compiled `adcprep`
  --aswip-executable ASWIP_EXECUTABLE
                        filename of compiled `aswip`
  --adcirc-processors ADCIRC_PROCESSORS
                        numbers of processors to assign for ADCIRC
  --job-duration JOB_DURATION
                        wall clock time for job
  --output-directory OUTPUT_DIRECTORY
                        directory to which to write configuration files (defaults to `.`)
  --skip-existing       skip existing files
  --absolute-paths      write paths as absolute in configuration
  --verbose             show more verbose log messages
  --perturbations PERTURBATIONS
  --variables VARIABLES
```

### Make Storm Ensemble (`make_storm_ensemble`)

```
usage: make_storm_ensemble [-h]
                           number-of-perturbations storm-code [start-date] [end-date] [file-deck] [mode] [record-type]
                           [directory]

positional arguments:
  number-of-perturbations
                        number of perturbations
  storm-code            storm name/code
  start-date            start date
  end-date              end date
  file-deck             letter of file deck, one of `a`, `b`
  mode                  either `realtime` / `aid_public` or `historical` / `archive`
  record-type           record type (i.e. `BEST`, `OFCL`)
  directory             output directory

optional arguments:
  -h, --help            show this help message and exit
```
