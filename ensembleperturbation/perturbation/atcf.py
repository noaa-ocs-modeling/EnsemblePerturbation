from abc import ABC
import concurrent.futures
from collections import namedtuple
from copy import copy
from difflib import get_close_matches
from datetime import datetime, timedelta
from enum import Enum, Flag, auto
from glob import glob
import json
from math import exp, inf, sqrt
import os
from os import PathLike
from pathlib import Path
from random import gauss, uniform
from typing import Dict, List, Mapping, Union, Optional
import warnings

import chaospy
from chaospy import Distribution
from dateutil.parser import parse as parse_date
import numpy
from numpy import floor, interp, sign
from pandas import DataFrame
from pandas.errors import SettingWithCopyWarning
import pint
from pint import Quantity, UnitStrippedWarning
from pint_pandas import PintArray, PintType
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection
from scipy.special import erfinv
from shapely.geometry import LineString
from stormevents.nhc import VortexTrack
from stormevents.nhc.atcf import ATCF_FileDeck
import typepigeon
import xarray
from xarray import Dataset

from ensembleperturbation.utilities import (
    get_logger,
    ProcessPoolExecutorStackTraced,
    units,
    move_to_end,
)

LOGGER = get_logger('perturbation.atcf')

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UnitStrippedWarning)

AIR_DENSITY = 1.15 * units.kilogram / units.meters ** 3

E1 = exp(1.0)  # e

BLAdj = 0.9  # adjustment factor from 10-m height to top of boundary layer

omega = 7.2921159 * 1e-5 / units.seconds  # angular velocity of the earth

# Index of absolute errors (forecast times [hrs)]
HISTORICAL_ERROR_HOURS = [0, 12, 24, 36, 48, 60, 72, 96, 120]  # has 60-hr data (for Rmax)
HISTORICAL_ERROR_HOURS_EXT = [
    0,
    12,
    24,
    36,
    48,
    60,
    72,
    84,
    96,
    108,
    120,
]  # extended index for new Rmax errors
HISTORICAL_ERROR_HOURS_NO_60H = (
    HISTORICAL_ERROR_HOURS[:5] + HISTORICAL_ERROR_HOURS[6:]
)  # no 60-hr data


class PerturbationType(Enum):
    GAUSSIAN = 'gaussian'
    UNIFORM = 'uniform'


class VortexVariable(ABC):
    name: str

    def __init__(
        self, default: float = None, unit: pint.Unit = None,
    ):
        self.__unit = None
        self.__default = None

        self.unit = unit
        self.default = default

    @property
    def unit(self) -> pint.Unit:
        return self.__unit

    @unit.setter
    def unit(self, unit: Union[str, pint.Unit]):
        if not isinstance(unit, pint.Unit):
            if unit is None:
                unit = ''
            unit = units.Unit(unit)
        self.__unit = unit

    @property
    def default(self) -> Quantity:
        if self.__default is not None and self.__default.units != self.unit:
            self.__default.ito(self.unit)
        return self.__default

    @default.setter
    def default(self, default: float):
        if isinstance(default, Quantity):
            if default.units != self.unit:
                default = default.to(self.unit)
        elif isinstance(default, tuple):
            units.Quantity.from_tuple(default)
        elif default is not None:
            default *= self.unit
        self.__default = default

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lower_bound={repr(self.lower_bound)}, upper_bound={repr(self.upper_bound)}, historical_forecast_errors={repr(self.historical_forecast_errors)}, default={repr(self.default)}, unit={repr(self.unit)})'


class CentralPressure(VortexVariable):
    name = 'central_pressure'
    unit = units.millibar


class BackgroundPressure(VortexVariable):
    name = 'background_pressure'

    def __init__(self):
        super().__init__(
            default=1013.0, unit=units.millibar,
        )


class VortexPerturbedVariable(VortexVariable, ABC):
    perturbation_type: PerturbationType

    def __init__(
        self,
        lower_bound: float = None,
        upper_bound: float = None,
        historical_forecast_errors: Dict[str, DataFrame] = None,
        default: float = None,
        unit: pint.Unit = None,
    ):
        super().__init__(default=default, unit=unit)

        self.__lower_bound = None
        self.__upper_bound = None
        self.__historical_forecast_errors = None

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.historical_forecast_errors = historical_forecast_errors

    @property
    def lower_bound(self) -> Quantity:
        if self.__lower_bound.units != self.unit:
            self.__lower_bound.ito(self.unit)
        return self.__lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound: float):
        if isinstance(lower_bound, Quantity):
            if lower_bound.units != self.unit:
                lower_bound = lower_bound.to(self.unit)
        elif isinstance(lower_bound, tuple):
            units.Quantity.from_tuple(lower_bound)
        elif lower_bound is not None:
            lower_bound *= self.unit
        self.__lower_bound = lower_bound

    @property
    def upper_bound(self) -> Quantity:
        if self.__upper_bound.units != self.unit:
            self.__upper_bound.ito(self.unit)
        return self.__upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound: float):
        if isinstance(upper_bound, Quantity):
            if upper_bound.units != self.unit:
                upper_bound = upper_bound.to(self.unit)
        elif isinstance(upper_bound, tuple):
            units.Quantity.from_tuple(upper_bound)
        elif upper_bound is not None:
            upper_bound *= self.unit
        self.__upper_bound = upper_bound

    @property
    def historical_forecast_errors(self) -> Dict[str, DataFrame]:
        for classification, dataframe in self.__historical_forecast_errors.items():
            for column in dataframe:
                pint_type = PintType(self.unit)
                if (
                    not isinstance(dataframe[column].dtype, PintType)
                    or dataframe[column].dtype != pint_type
                ):
                    if (
                        isinstance(dataframe[column].dtype, PintType)
                        and dataframe[column].dtype != pint_type
                    ):
                        dataframe[column] = dataframe[column].pint.to(self.unit)
                    dataframe[column].astype(pint_type, copy=False)
        return self.__historical_forecast_errors

    @historical_forecast_errors.setter
    def historical_forecast_errors(self, historical_forecast_errors: Dict[str, DataFrame]):
        for classification, dataframe in historical_forecast_errors.items():
            for column in dataframe:
                pint_type = PintType(self.unit)
                if (
                    not isinstance(dataframe[column].dtype, PintType)
                    or dataframe[column].dtype != pint_type
                ):
                    if (
                        isinstance(dataframe[column].dtype, PintType)
                        and dataframe[column].dtype != pint_type
                    ):
                        dataframe[column] = dataframe[column].pint.to(self.unit)
                    dataframe[column].astype(pint_type, copy=False)
        self.__historical_forecast_errors = historical_forecast_errors

    def chaospy_distribution(self) -> chaospy.Distribution:
        # TODO see if we need to transform these distributions into unit space
        # TODO figure out how to account for piecewise uncertainty

        if self.perturbation_type == PerturbationType.GAUSSIAN:
            distribution = chaospy.Normal(mu=0, sigma=1)
        elif self.perturbation_type == PerturbationType.UNIFORM:
            distribution = chaospy.Uniform(lower=-1, upper=1)
        else:
            raise ValueError(f'perturbation type {self.perturbation_type} not recognized')

        return distribution

    def perturb(
        self,
        vortex_dataframe: DataFrame,
        values: List[float],
        times: List[datetime],
        inplace: bool = False,
    ) -> DataFrame:
        """
        perturb the variable within physical bounds

        :param vortex_dataframe: ATCF dataframe containing track info
        :param values: values for each forecast time (VT)
        :param times: forecast times (VT)
        :param inplace: modify dataframe in-place
        :return: updated ATCF dataframe with perturbed values
        """

        if not inplace:
            # make a deepcopy to preserve the original dataframe
            vortex_dataframe = vortex_dataframe.copy(deep=True)

        if self.name in vortex_dataframe.columns:
            varname = self.name
        else:
            varname = get_close_matches(self.name, vortex_dataframe.columns)[0]
        variable_values = vortex_dataframe[varname].values
        if (
            not isinstance(variable_values, PintArray)
            or variable_values.units == variable_values.units._REGISTRY.dimensionless
        ):
            variable_values *= self.unit
        if isinstance(variable_values, PintArray):
            variable_values = variable_values.quantity
        if (
            not isinstance(values, Quantity)
            or values.units == values.units._REGISTRY.dimensionless
        ):
            values *= self.unit

        all_values = variable_values - values
        # ensure that we don't change zero values
        all_values[variable_values.magnitude < 1] = 0 * all_values.units
        vortex_dataframe[varname] = [
            min(self.upper_bound, max(value, self.lower_bound)).magnitude
            if value.magnitude != 0
            else 0
            for value in all_values
        ] * self.unit

        return vortex_dataframe

    def storm_errors(self, data_frame: DataFrame, loc_index: int = 0) -> DataFrame:
        """
        Historical forecast errors of the given storm, based on initial intensity.

        :param data_frame: storm data frame
        :param loc_index: index to classify the errors at (0 by default for initial)
        :return: errors based on intensity classification
        """

        initial_intensity = data_frame[MaximumSustainedWindSpeed.name].iloc[loc_index]

        if not isinstance(initial_intensity, Quantity):
            initial_intensity *= units.knot

        if initial_intensity < 50 * units.knot:
            storm_classification = '<50kt'  # weak
        elif initial_intensity <= 95 * units.knot:
            storm_classification = '50-95kt'  # medium
        else:
            storm_classification = '>95kt'  # strong

        LOGGER.debug(f'storm classification: {storm_classification}')
        return self.historical_forecast_errors[storm_classification]


class MaximumSustainedWindSpeed(VortexPerturbedVariable):
    """
    ``max_sustained_wind_speed`` (``Vmax``) represents the maximum wind speed sustained by the storm.
    It is perturbed along a random gaussian distribution (``0`` - ``1``), scaled to the mean of absolute historical errors.
    ``central_pressure`` (``pc``) is then changed proportionally based on the Holland B parameter.
    """

    name = 'max_sustained_wind_speed'
    perturbation_type = PerturbationType.GAUSSIAN
    unit = units.knot

    # Reference - 2019_Psurge_Error_Update_FINAL.docx
    # Table 12: Adjusted intensity errors [kt] for 2015-2019
    def __init__(self):
        super().__init__(
            lower_bound=15,
            upper_bound=175,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {'mean error [kt]': [1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91]},
                    dtype=PintType(units.knot),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {'mean error [kt]': [2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62]},
                    dtype=PintType(units.knot),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '>95kt': DataFrame(
                    {
                        'mean error [kt]': [
                            2.80,
                            7.94,
                            11.53,
                            13.27,
                            12.66,
                            13.41,
                            13.46,
                            13.55,
                        ]
                    },
                    dtype=PintType(units.knot),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
            },
            unit=units.knot,
        )

    def radial_Vmax_at_BL(self, dataframe: DataFrame) -> Quantity:
        # keep original translation speed for the 0-hr forecast which has been computed
        # using info from track info prior to the forecast time (in m/s!)
        initial_translation_speed = dataframe['speed'].loc[
            dataframe['datetime'] == dataframe['datetime'].min()
        ].values * units('m/s')
        # re-compute the translation speed
        translation_speed = VortexTrack._VortexTrack__compute_velocity(dataframe)[
            'speed'
        ].values * units('m/s')
        # overwrite the first hour with the initial
        translation_speed[
            dataframe['datetime'] == dataframe['datetime'].min()
        ] = initial_translation_speed
        # make sure put this back into the dataframe to preserve
        dataframe['speed'] = translation_speed.magnitude
        # get the radial Vmax at the boundary layer height
        SWH79_ts = 1.5 * (translation_speed.magnitude ** 0.63) * (0.514791 ** 0.37)
        Vmax = (
            dataframe[self.name].values * self.unit
            - SWH79_ts * translation_speed.units  # SHW79 eqn
            # - 0.62 * translation_speed  # LC12
        ) / BLAdj
        # limit Vmax to the lower bound
        Vmax[Vmax < self.lower_bound] = self.lower_bound
        return Vmax, translation_speed


class RadiusOfMaximumWinds(VortexPerturbedVariable):
    """
    ``radius_of_maximum_winds`` (``Rmax``) represents the distance from the storm center to the maximum wind speed sustained by the storm.
    It is perturbed along a random gaussian distribution (``0`` - ``1``), scaled to the mean of absolute historical errors based on the forecasted Rmax regression in Penny et al. (2023). https://doi.org/10.1175/WAF-D-22-0209.1
    """

    name = 'radius_of_maximum_winds'
    perturbation_type = PerturbationType.GAUSSIAN
    unit = units.nautical_mile

    # Reference - Computed MAE from the CDF of 10-yrs of RMW forecast errors as described in Penny et al. (2023), and provided by Andrew Penny.
    def __init__(self):
        super().__init__(
            lower_bound=5,
            upper_bound=200,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {
                        'mean error [nmi]': [
                            3.38,
                            11.81,
                            15.04,
                            17.44,
                            17.78,
                            17.78,
                            18.84,
                            19.34,
                            21.15,
                            23.19,
                            24.77,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_EXT,
                ),
                '50-95kt': DataFrame(
                    {
                        'mean error [nmi]': [
                            3.38,
                            6.67,
                            9.51,
                            10.70,
                            11.37,
                            12.27,
                            12.67,
                            13.32,
                            13.97,
                            14.45,
                            15.79,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_EXT,
                ),
                '>95kt': DataFrame(
                    {
                        'mean error [nmi]': [
                            3.38,
                            3.66,
                            4.98,
                            6.26,
                            7.25,
                            8.57,
                            9.22,
                            10.69,
                            11.35,
                            11.57,
                            11.38,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_EXT,
                ),
            },
            unit=units.nautical_mile,
        )


class RadiusOfMaximumWindsPersistent(VortexPerturbedVariable):
    """
    ``radius_of_maximum_winds_persistent`` (``Rmax``)
    It is perturbed along a uniform distribution on [-1,1] between the 0th and 100th percentile CDFs of NHC historical forecast errors based on persistence forecasting of Rmax (0-hr Rmax is propagated for every validation time of storm).
    0th and 100th percentiles are calculated based on extrapolation from the provided 15th, 50th, and 85th percentiles.
    """

    name = 'radius_of_maximum_winds_persistent'
    perturbation_type = PerturbationType.UNIFORM

    def __init__(self):
        super().__init__(
            lower_bound=5,
            upper_bound=200,
            historical_forecast_errors={
                '<15sm': DataFrame(
                    {
                        '15th percentile error': [
                            2 * -13.82 - -19.67,
                            -13.82,
                            -19.67,
                            -21.37,
                            -26.31,
                            -32.71,
                            -39.12,
                            -46.80,
                            -52.68,
                        ],
                        '50th percentile error': [
                            2 * -2.72 - -6.74,
                            -2.72,
                            -6.74,
                            -9.59,
                            -12.12,
                            -15.64,
                            -19.16,
                            -18.60,
                            -24.07,
                        ],
                        '85th percentile error': [
                            2 * 1.27 - 0.22,
                            1.27,
                            0.22,
                            1.02,
                            0.00,
                            -2.59,
                            -5.18,
                            -7.15,
                            -12.91,
                        ],
                    },
                    dtype=PintType(units.us_statute_mile),
                    index=HISTORICAL_ERROR_HOURS,
                ),
                '15-25sm': DataFrame(
                    {
                        '15th percentile error': [
                            2 * -10.47 - -14.54,
                            -10.47,
                            -14.54,
                            -20.35,
                            -23.88,
                            -21.78,
                            -19.68,
                            -24.24,
                            -28.30,
                        ],
                        '50th percentile error': [
                            2 * -1.07 - -2.48,
                            -1.07,
                            -2.48,
                            -4.25,
                            -4.33,
                            -3.55,
                            -2.77,
                            -6.02,
                            -4.44,
                        ],
                        '85th percentile error': [
                            2 * 4.17 - 6.70,
                            4.17,
                            6.70,
                            6.13,
                            6.54,
                            6.93,
                            7.32,
                            9.33,
                            8.03,
                        ],
                    },
                    dtype=PintType(units.us_statute_mile),
                    index=HISTORICAL_ERROR_HOURS,
                ),
                '25-35sm': DataFrame(
                    {
                        '15th percentile error': [
                            2 * -8.57 - -13.41,
                            -8.57,
                            -13.41,
                            -10.87,
                            -9.26,
                            -9.34,
                            -9.42,
                            -7.41,
                            -7.40,
                        ],
                        '50th percentile error': [
                            2 * 0.39 - 1.66,
                            0.39,
                            1.66,
                            2.49,
                            4.42,
                            5.20,
                            5.98,
                            5.98,
                            6.96,
                        ],
                        '85th percentile error': [
                            2 * 8.21 - 10.62,
                            8.21,
                            10.62,
                            13.93,
                            15.62,
                            16.04,
                            16.46,
                            16.51,
                            16.70,
                        ],
                    },
                    dtype=PintType(units.us_statute_mile),
                    index=HISTORICAL_ERROR_HOURS,
                ),
                '35-45sm': DataFrame(
                    {
                        '15th percentile error': [
                            2 * -10.66 - -7.64,
                            -10.66,
                            -7.64,
                            -5.68,
                            -3.25,
                            -1.72,
                            -0.19,
                            3.65,
                            2.59,
                        ],
                        '50th percentile error': [
                            2 * 3.22 - 7.32,
                            3.22,
                            7.32,
                            12.67,
                            14.14,
                            13.72,
                            13.31,
                            14.60,
                            14.01,
                        ],
                        '85th percentile error': [
                            2 * 14.77 - 17.85,
                            14.77,
                            17.85,
                            22.07,
                            27.60,
                            27.08,
                            26.56,
                            26.80,
                            28.30,
                        ],
                    },
                    dtype=PintType(units.us_statute_mile),
                    index=HISTORICAL_ERROR_HOURS,
                ),
                '>45sm': DataFrame(
                    {
                        '15th percentile error': [
                            2 * -15.36 - -10.37,
                            -15.36,
                            -10.37,
                            3.14,
                            12.10,
                            12.21,
                            12.33,
                            6.66,
                            7.19,
                        ],
                        '50th percentile error': [
                            2 * 8.11 - 15.19,
                            8.11,
                            15.19,
                            17.21,
                            24.25,
                            25.63,
                            27.00,
                            20.56,
                            20.51,
                        ],
                        '85th percentile error': [
                            2 * 21.43 - 29.96,
                            21.43,
                            29.96,
                            37.22,
                            39.27,
                            39.10,
                            38.93,
                            34.40,
                            35.93,
                        ],
                    },
                    dtype=PintType(units.us_statute_mile),
                    index=HISTORICAL_ERROR_HOURS,
                ),
            },
            unit=units.nautical_mile,
        )

    def storm_errors(self, data_frame: DataFrame, loc_index: int = 0) -> DataFrame:
        """
        Historical forecast errors of the given storm, based on initial size.

        :param data_frame: storm data frame
        :param loc_index: index to classify the errors at (0 by default for initial)
        :return: errors based on size classification
        """

        initial_radius = data_frame['radius_of_maximum_winds'].iloc[loc_index]

        if not isinstance(initial_radius, Quantity):
            initial_radius *= units.nautical_mile

        if initial_radius < 15 * units.us_statute_mile:
            storm_classification = '<15sm'  # very small
        elif initial_radius < 25 * units.us_statute_mile:
            storm_classification = '15-25sm'  # small
        elif initial_radius < 35 * units.us_statute_mile:
            storm_classification = '25-35sm'  # medium
        elif initial_radius <= 45 * units.us_statute_mile:
            storm_classification = '35-45sm'  # large
        else:
            storm_classification = '>45sm'  # very large

        LOGGER.debug(f'storm classification: {storm_classification}')
        return self.historical_forecast_errors[storm_classification]


IsotachQuadrant = namedtuple('IsotachQuadrant', ['angle', 'column'])


class IsotachRadiusQuadrant(Enum):
    SWQ = IsotachQuadrant(225, 'isotach_radius_for_SWQ')
    SEQ = IsotachQuadrant(315, 'isotach_radius_for_SEQ')
    NWQ = IsotachQuadrant(135, 'isotach_radius_for_NWQ')
    NEQ = IsotachQuadrant(45, 'isotach_radius_for_NEQ')


# IsotachRadius adjuster dependent on the changes to Rmax, Vmax and track
class IsotachRadius:
    def __init__(self, rad_quad_info: IsotachRadiusQuadrant):
        self.quadrant_angle = rad_quad_info.value.angle
        self.column = rad_quad_info.value.column

    def find_GAHM_parameters(self, B, Ro_inv) -> Quantity:
        """ 
        find Bg and phi for GAHM model iteratively
        with given B and Rossby number (inverse)
        """
        Bg = B
        Bm = 1e6
        while any(abs(Bm - Bg) > 1e-3):
            Bm = Bg
            phi = 1 + Ro_inv / (Bg * (1 + Ro_inv))
            Bg = B * (1 + Ro_inv) * numpy.exp(phi - 1) / phi
        return Bg, phi

    def find_parameter_from_GAHM_profile(
        self,
        Vr,
        Vmax,
        f,
        Ro_inv,
        B=None,
        Bg=None,
        phi=None,
        Rmax=None,
        isotach_rad=None,
        Rrat=None,
    ) -> Quantity:
        """ 
        Use root finding algorithm to return a desired parameter 
        based on the GAHM profile of velocity that matches the input Vr
        Input B (guess), isotach_rad and Rrat to get back the actual B
        Input Bg, phi, Rmax, and Rrat to get back isotach_rad 
        """
        if B is not None:
            # initial guesses when trying to find Bg, phi (and new B)
            Bg, phi = self.find_GAHM_parameters(B, Ro_inv)
            rfo2 = 0.5 * isotach_rad * f  # this stays constant
        else:
            # updates for when trying to find isotach_rad
            rfo2 = 0.5 * Rmax * f  # this would be rfo2 at Rrat = alpha = 1
        alpha = Rrat ** Bg
        alpha_lo = numpy.nan * alpha
        alpha_hi = 0 * alpha + 1
        Vr_test = 1e6 * MaximumSustainedWindSpeed.unit
        tol = 1e-2 * MaximumSustainedWindSpeed.unit
        beta = Vmax ** 2 * (1 + Ro_inv)
        beta[
            numpy.sqrt(beta + rfo2 ** 2) - rfo2 < Vr - 0.5 * tol
        ] = numpy.nan  # no possible solution
        i = 0
        itmax = 1000
        while any(abs(Vr_test - Vr) > tol):
            # updates to desired parameters with current alpha value
            if B is not None:
                # update to Bg, phi
                Bg = numpy.log(alpha) / numpy.log(Rrat)
                phi = 1 + Ro_inv / (Bg * (1 + Ro_inv))
                phi[phi < 1] = 1
            else:
                # update to Rrat
                Rrat = alpha ** (1 / Bg)
                isotach_rad = Rmax / Rrat
                rfo2 = 0.5 * isotach_rad * f
            # compute Vr using the current set of inputs
            expf = numpy.exp(phi * (1 - alpha))
            Vr_test = numpy.sqrt(beta * expf * alpha + rfo2 ** 2) - rfo2
            # updates to the alphas for bi-section method
            alpha_hi[Vr_test > Vr] = alpha[Vr_test > Vr]
            alpha_lo[Vr_test < Vr] = alpha[Vr_test < Vr]
            # guess new alpha based on error
            alpha[Rrat <= 1] *= (Vr / Vr_test)[Rrat <= 1] ** 2
            alpha[Rrat > 1] *= (Vr_test / Vr)[Rrat > 1] ** 2
            # bi-section method to help convergence
            avail = ~numpy.isnan(alpha_lo)
            alpha[avail] = 0.5 * (alpha_hi[avail] + alpha_lo[avail])
            i += 1
            if i == itmax:
                raise RuntimeError('GAHM function could not converge')
        if B is not None:
            return Bg * phi / ((1 + Ro_inv) * numpy.exp(phi - 1))  # B
        else:
            return Rmax / Rrat  # isotach_rad

    def get_isotach_properties(self, dataframe: DataFrame) -> Quantity:
        """ 
        Return properties of storm and isotach after adjusting for quadrant angle and boundary layer height.

        LC12: Lin, N., and D. Chavas (2012). On hurricane parametric wind and applications in storm surge modeling,
        J. Geophys. Res., 117, D09120, doi:10.1029/2011JD017126

        SHW79: Schwerdt, R. W., Ho, F. P., & Watkins, R. R. (1979). Meteorological Criteria for Standard Project Hurricane and Probable Maximum Hurricane Windfields, Gulf and East Coasts of the United States.
        https://repository.library.noaa.gov/view/noaa/6948/noaa_6948_DS1.pdf
        """

        def ts_factor(isotach_rad, Rmax):
            # factors based on LC12 Figure 2
            rRm = isotach_rad / Rmax
            rRm[rRm == 0] = 1e6  # convert a zero value to the upper limit
            factor = 0.66 + (0.6 - 0.66) * (rRm - 1)  # (1 <= rRm <= 2)
            factor[rRm > 2] = 0.6 + (0.55 - 0.6) * (rRm[rRm > 2] - 2) / 4
            factor[rRm > 6] = 0.55
            factor[rRm < 1] = 0.3 + (0.66 - 0.3) * rRm[rRm < 1]
            return factor

        # get Vmax after subtracting speed and adjusting to boundary layer height
        Vmax, translation_speed = MaximumSustainedWindSpeed().radial_Vmax_at_BL(dataframe)
        # get Vr for the X-kt isotach in this quadrant
        SWH79_ts = 1.5 * (translation_speed.magnitude ** 0.63) * (0.514791 ** 0.37)
        # find the ts rotation angle based on the radius weighted sum
        isotach_complex = dataframe.filter(regex='isotach_radius_for').values * [
            1 + 1j,
            1 - 1j,
            -1 - 1j,
            -1 + 1j,
        ]
        ts_rotation = numpy.rad2deg(numpy.angle(isotach_complex.sum(axis=1)))
        Vr = (
            dataframe['isotach_radius'].values * MaximumSustainedWindSpeed.unit
            - SWH79_ts
            * translation_speed.units
            * numpy.cos(
                numpy.deg2rad(self.quadrant_angle - ts_rotation)
            )  # SHW79 method assuming inflow angle = 25-deg @r & 10-deg @RMW, i.e., 15-deg diff
            # although SHW79 suggest maximum is in SEQ, here we assume it is in NEQ and use minus 15 deg
            # since that appears to match the NHC data better (also more similar to LC12)
            # LC12 factor and 20-deg CCW rotation
            # - ts_factor(
            #    dataframe[self.column].values, dataframe[RadiusOfMaximumWinds.name].values
            # )
            # * translation_speed
            # * numpy.sin(numpy.deg2rad(self.quadrant_angle + 20))
        ) / BLAdj
        # 0-kt isotach's should be ignored
        Vr[dataframe['isotach_radius'].values == 0] = numpy.nan
        # get Rmax with units
        Rmax = dataframe[RadiusOfMaximumWinds.name].values * RadiusOfMaximumWinds.unit
        # get Coriolis and compute Rossby number (inverse of)
        f = 2 * omega * numpy.sin(numpy.deg2rad(dataframe['latitude'].values))
        Ro_inv = (Rmax * f / Vmax).to('dimensionless')
        # get central pessure and compute standard Holland B parameter as initial guess
        DelP = dataframe[BackgroundPressure.name] - dataframe[CentralPressure.name]
        DelP = DelP.values * CentralPressure.unit
        B = Vmax.to('m/s') ** 2 * AIR_DENSITY * E1 / DelP.to('Pa')
        # now found the Bg and phi (and subsequent B) that actually fits
        # through both the isotach and the Rmax value
        return Vmax, Vr, Rmax, Ro_inv, f, B

    def adjust(
        self, vortex_dataframe: DataFrame, original_dataframe, inplace: bool = False,
    ) -> DataFrame:
        """ Compute new isotach radius based on Holland profile using perturbed Rmax/Vmax/speed values"""
        # retrieve the Vmax, Vr(isotach) at the top of the boundary layer, and Rmax
        # in the original and perturbed dataframe
        Vmax_old, Vr_old, Rmax_old, Roinv_old, f_old, B_ini = self.get_isotach_properties(
            dataframe=original_dataframe
        )
        Vmax_new, Vr_new, Rmax_new, Roinv_new, f_new, _ = self.get_isotach_properties(
            dataframe=vortex_dataframe
        )
        # determine original isotach_rad and Rrat
        isotach_rad = original_dataframe[self.column].values * RadiusOfMaximumWinds.unit
        # fill in isotach_rad==0 using Rankin vortex assumption
        # for the mean isotach radius (calculated across non-zero quadrants)
        isotach_radall = original_dataframe.filter(regex='isotach_radius_for').values
        isotach_radall[isotach_radall == 0] = numpy.nan
        radmean = numpy.nanmean(isotach_radall, axis=1) * isotach_rad.units
        isotach_rad_adjust = (
            radmean * original_dataframe['isotach_radius'].values / Vr_old.magnitude
        )
        isotach_rad[isotach_rad == 0] = isotach_rad_adjust[isotach_rad == 0]
        Rrat = Rmax_old / isotach_rad  # [nmi/nmi] = []
        # correct where Rrat is >= 1
        invalid = Rrat >= 1
        Rrat[invalid] = Vr_old[invalid] / Vmax_old[invalid]  # Rankine vortex assumption
        # find B from original velocity profile using GAHM (traditional Holland B as first guess)
        B = self.find_parameter_from_GAHM_profile(
            Vr_old, Vmax_old, f_old, Roinv_old, B=B_ini, isotach_rad=isotach_rad, Rrat=Rrat
        )
        # correct where B is nan
        invalid = numpy.isnan(B)
        B[invalid] = B_ini[invalid]
        # preserving B, find new GAHM Bg and phi using new Ro_inv
        Bg, phi = self.find_GAHM_parameters(B, Roinv_new)
        # determine guess of new Rrat
        Rrat = Rmax_new / isotach_rad
        # correct where Rrat is nan or >= 1
        invalid = (numpy.isnan(Rrat)) | (Rrat >= 1)
        Rrat[invalid] = Vr_new[invalid] / Vmax_new[invalid]  # Rankine vortex assumption
        Rrat[Vr_new > Vmax_new] = numpy.nan  # if Vr is stronger than Vmax then ignore
        # now use GAHM root finding algorithm to get the new isotach_rad
        isotach_rad_new = self.find_parameter_from_GAHM_profile(
            Vr_new, Vmax_new, f_new, Roinv_new, Bg=Bg, phi=phi, Rmax=Rmax_new, Rrat=Rrat,
        )
        isotach_rad_new = isotach_rad_new.magnitude
        isotach_rad_new[numpy.isnan(isotach_rad_new)] = 0
        vortex_dataframe[self.column] = isotach_rad_new
        return vortex_dataframe


class CrossTrack(VortexPerturbedVariable):
    """
    ``cross_track``  represents a perpendicular offset of the tropical cyclone center track, accomplished by moving each forecast time left or right perpedicular to the track line.
    It is perturbed along a random gaussian distribution (``0`` - ``1``), scaled to the mean of absolute historical errors.
    """

    name = 'cross_track'
    perturbation_type = PerturbationType.GAUSSIAN

    # Reference - 2019_Psurge_Error_Update_FINAL.docx
    # Table 8: Adjusted cross-track errors [nm] for 2015-2019
    def __init__(self):
        super().__init__(
            lower_bound=-inf,
            upper_bound=+inf,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {
                        'mean error [nm]': [
                            4.98,
                            16.16,
                            23.10,
                            28.95,
                            38.03,
                            56.88,
                            92.95,
                            119.67,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {
                        'mean error [nm]': [
                            2.89,
                            11.58,
                            16.83,
                            21.10,
                            27.76,
                            47.51,
                            68.61,
                            103.45,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '>95kt': DataFrame(
                    {
                        'mean error [nm]': [
                            1.85,
                            7.79,
                            12.68,
                            17.92,
                            25.01,
                            40.48,
                            60.69,
                            79.98,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
            },
            unit=units.nautical_mile,
        )

    def perturb(
        self,
        vortex_dataframe: DataFrame,
        values: List[float],
        times: List[datetime],
        inplace: bool = False,
    ) -> DataFrame:
        """
        offsets points by a given perpendicular error/distance from the original track

        :param vortex_dataframe: ATCF dataframe containing track info
        :param values: cross-track errors [nm] for each forecast time (VT)
        :param times: forecast times (VT)
        :param inplace: modify dataframe in-place
        :return: updated ATCF dataframe with different longitude latitude locations based on perpendicular offsets set by the cross_track_errors
        """

        if not inplace:
            # make a deepcopy to preserve the original dataframe
            vortex_dataframe = vortex_dataframe.copy(deep=True)

        # Get the coordinates of the track
        points = vortex_dataframe[['longitude', 'latitude']].values

        # set the EPSG of the track coordinates
        wgs84 = CRS.from_epsg(4326)

        times = (times / timedelta(hours=1)).values * units.hours

        # loop over all coordinates
        new_coordinates = []
        for current_index in range(0, len(points)):
            current_point = points[current_index]

            # get the utm projection for the reference coordinate
            utm_crs = utm_crs_from_longitude(current_point[0])
            transformer = Transformer.from_crs(wgs84, utm_crs)

            # get the current cross_track_error
            cross_track_error = values[current_index]

            # get the location of the original reference coordinate
            current_point = (
                transformer.transform(current_point[1], current_point[0]) * units.meter
            )

            # get the index of the previous forecasted coordinate
            previous_index = current_index - 1
            while previous_index >= 0:
                if times[previous_index] < times[current_index]:
                    break
                previous_index = previous_index - 1
            if previous_index < 0:  # beginning of track
                previous_index = current_index

            # get previous projected coordinate
            previous_point = points[previous_index]
            previous_point = (
                transformer.transform(previous_point[1], previous_point[0]) * units.meter
            )

            # get the perpendicular offset based on the line connecting from the previous coordinate to the current coordinate
            previous_offset = get_offset(previous_point, current_point, cross_track_error)

            # get the index of the next forecasted coordinate
            next_index = current_index + 1
            while next_index < len(points):
                if times[next_index] > times[current_index]:
                    break
                next_index = next_index + 1
            if next_index == len(points):  # end of track
                next_index = current_index

            # get previous projected coordinate
            next_point = points[next_index]
            next_point = transformer.transform(next_point[1], next_point[0]) * units.meter

            # get the perpendicular offset based on the line connecting from the current coordinate to the next coordinate
            next_offset = get_offset(current_point, next_point, cross_track_error)

            # get the perpendicular offset based on the average of the forward and backward piecewise track lines adjusted so that the distance matches the actual cross_error
            normal_offset = numpy.mean([previous_offset, next_offset], axis=0)
            alpha = abs(cross_track_error) / numpy.sqrt(numpy.sum(normal_offset ** 2))

            if numpy.isinf(alpha):
                alpha = 0

            # compute the next point and retrieve back the lat-lon geographic coordinate
            new_point = current_point - alpha * normal_offset
            new_coordinates.append(
                transformer.transform(
                    new_point[0].magnitude,
                    new_point[1].magnitude,
                    direction=TransformDirection.INVERSE,
                )
            )

        vortex_dataframe['latitude'], vortex_dataframe['longitude'] = zip(*new_coordinates)

        return vortex_dataframe


class AlongTrack(VortexPerturbedVariable):
    """
    ``along_track`` represents a parallel offset of the tropical cyclone center track, accomplished by moving each forecast time forward or backward along the track line.
    It is perturbed along a random gaussian distribution (``0`` - ``1``), scaled to the mean of absolute historical errors.
    """

    name = 'along_track'
    perturbation_type = PerturbationType.GAUSSIAN

    # Reference - 2019_Psurge_Error_Update_FINAL.docx
    # Table 7: Adjusted along-track errors [nm] for 2015-2019
    def __init__(self):
        super().__init__(
            lower_bound=-inf,
            upper_bound=+inf,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {
                        'mean error [nm]': [
                            6.33,
                            17.77,
                            26.66,
                            37.75,
                            51.07,
                            69.22,
                            108.59,
                            125.01,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {
                        'mean error [nm]': [
                            3.68,
                            12.74,
                            19.43,
                            27.51,
                            37.28,
                            57.82,
                            80.15,
                            108.07,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '>95kt': DataFrame(
                    {
                        'mean error [nm]': [
                            2.35,
                            8.57,
                            14.64,
                            23.36,
                            33.59,
                            49.26,
                            70.90,
                            83.55,
                        ]
                    },
                    dtype=PintType(units.nautical_mile),
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
            },
            unit=units.nautical_mile,
        )

    def perturb(
        self,
        vortex_dataframe: DataFrame,
        values: List[float],
        times: List[datetime],
        inplace: bool = False,
    ) -> DataFrame:
        """
        offsets points by a given error/distance by interpolating along the track

        :param vortex_dataframe: ATCF dataframe containing track info
        :param values: along-track errors for each forecast time (VT)
        :param times: forecast timed (VT)
        :param inplace: modify dataframe in-place
        :return: updated ATCF dataframe with different longitude latitude locations based on interpolated errors along track
        """

        if not inplace:
            # make a deepcopy to preserve the original dataframe
            vortex_dataframe = vortex_dataframe.copy(deep=True)

        max_interpolated_points = 5  # maximum number of pts along line for each interpolation

        # Get the coordinates of the track
        coordinates = vortex_dataframe[['longitude', 'latitude']].values

        # set the EPSG of the track coordinates
        wgs84 = CRS.from_epsg(4326)

        hours = (times / timedelta(hours=1)).values

        unique_points, unique_indices = numpy.unique(coordinates, axis=0, return_index=True)
        unique_points = unique_points[numpy.argsort(unique_indices)]
        unique_times, unique_indices = numpy.unique(hours, axis=0, return_index=True)
        unique_times = unique_times[numpy.argsort(unique_indices)]

        # Extrapolating the track for negative errors at beginning and positive errors at end of track
        previous_diffs = numpy.flip(
            numpy.repeat(
                [unique_points[1] - unique_points[0]], max_interpolated_points, axis=0
            )
            * numpy.expand_dims(numpy.arange(1, max_interpolated_points + 1), axis=1),
            axis=0,
        )
        after_diffs = numpy.repeat(
            [unique_points[-1] - unique_points[-2]], max_interpolated_points, axis=0
        ) * numpy.expand_dims(numpy.arange(1, max_interpolated_points + 1), axis=1)
        coordinates = numpy.concatenate(
            [coordinates[0] - previous_diffs, coordinates, coordinates[-1] + after_diffs]
        )

        # adding pseudo-VT times to the ends
        previous_diffs = numpy.flip(
            numpy.repeat([unique_times[1] - unique_times[0]], max_interpolated_points)
            * numpy.arange(1, max_interpolated_points + 1)
        )
        after_diffs = numpy.repeat(
            [unique_times[-1] - unique_times[-2]], max_interpolated_points
        ) * numpy.arange(1, max_interpolated_points + 1)
        hours = (
            numpy.concatenate((hours[0] - previous_diffs, hours, hours[-1] + after_diffs))
            * units.hour
        )

        # loop over all coordinates
        new_coordinates = []
        # indices are the locations of extrapolated track
        for index in range(max_interpolated_points, len(hours) - max_interpolated_points):
            # get the utm projection for the reference coordinate
            utm_crs = utm_crs_from_longitude(coordinates[index][0])
            transformer = Transformer.from_crs(wgs84, utm_crs)

            along_error = -1.0 * values[index - max_interpolated_points]
            along_sign = int(sign(along_error))

            projected_points = []
            track_index = index
            while len(projected_points) < max_interpolated_points:
                if (
                    track_index == index
                    or hours[track_index] != hours[track_index - along_sign]
                ):
                    # get the x,y utm coordinate for this line string
                    projected_points.append(
                        transformer.transform(
                            coordinates[track_index][1], coordinates[track_index][0],
                        )
                    )
                track_index = track_index + along_sign

            # make the temporary line segment
            line_segment = LineString(projected_points)

            # interpolate a distance "along_error" along the line
            projected_coordinate = line_segment.interpolate(
                abs(along_error.to(units.meter).magnitude)
            )

            # get back lat-lon
            new_coordinates.append(
                transformer.transform(
                    projected_coordinate.coords[0][0],
                    projected_coordinate.coords[0][1],
                    direction=TransformDirection.INVERSE,
                )
            )

        vortex_dataframe['latitude'], vortex_dataframe['longitude'] = zip(*new_coordinates)

        return vortex_dataframe


class PerturberFeatures(Flag):
    NONE = 0
    ISOTACH_ADJUSTMENT = auto()


class VortexPerturber:
    """
    ``VortexPerturber`` takes an ATCF track from an input storm and perturbs it based on several variables (of the class ``VortexPerturbedVariable``)

    .. code-block:: python

        # retrieve initial storm track for Florence 2018 (defaults to archival best track)
        perturber = VortexPerturber(storm='florence2018')

        # write 3 tracks perturbed using specified perturbation values (perturbations are of sigma values (``0`` - ``1`` for uniform or ``-1`` - ``1`` for gaussian) that are then scaled to historical errors per-variable
        perturber.write(
            perturbations=[
                -1.0,
                {
                    MaximumSustainedWindSpeed: -0.25,
                    CrossTrack: 0.25,
                    'along_track': 0.75,
                    'radius_of_maximum_winds': -1,
                },
                0.75,
            ],
            variables=[MaximumSustainedWindSpeed, RadiusOfMaximumWinds, CrossTrack, AlongTrack],
            directory='./3_tracks_perturbed_specifically',
        )

        # write 5 randomly-perturbed tracks, drawing randomly from the distribution of each variable except for ``CrossTrack``
        perturber.write(
            perturbations=5,
            variables=[MaximumSustainedWindSpeed, RadiusOfMaximumWinds, AlongTrack],
            directory='./5_tracks_perturbed_randomly_except_crosstrack',
        )

        # write tracks perturbed along the quadrature (`4^n` where `n` is the number of variables)
        perturber.write(
            perturbations=None,
            quadrature=True,
            variables=[MaximumSustainedWindSpeed, RadiusOfMaximumWinds, CrossTrack, AlongTrack],
            directory='./256_tracks_perturbed_along_quadrature',
        )

    """

    def __init__(
        self,
        storm: str,
        start_date: datetime = None,
        end_date: datetime = None,
        file_deck: ATCF_FileDeck = None,
        advisories: List[str] = None,
        features: Optional[PerturberFeatures] = PerturberFeatures.ISOTACH_ADJUSTMENT,
    ):
        """
        :param storm: NHC storm code, for instance `al062018`
        :param start_date: start time of ensemble
        :param end_date: end time of ensemble
        :param file_deck: ATCF file deck; one of `a`, `b`, `f`
        :param advisories: ATCF advisory type; one of ``BEST``, ``OFCL``, ``OFCP``, ``HMON``, ``CARQ``, ``HWRF``
        """

        self.__start_date = None
        self.__end_date = None
        self.__file_deck = None
        self.__track = None
        self.__previous_configuration = None
        if features is None:
            features = PerturberFeatures.NONE
        self.__features = PerturberFeatures(features)

        self.storm = storm
        self.start_date = start_date
        self.end_date = end_date
        self.file_deck = file_deck
        self.advisories = advisories

        self.__filename = None

    @classmethod
    def from_file(
        cls,
        filename: PathLike,
        start_date: datetime = None,
        end_date: datetime = None,
        file_deck: ATCF_FileDeck = None,
        advisories: List[str] = None,
        features: Optional[PerturberFeatures] = PerturberFeatures.ISOTACH_ADJUSTMENT,
    ) -> 'VortexPerturber':
        """
        build storm perturber from an existing `fort.22` or ATCF file

        :param filename: file path to `fort.22` / ATCF file
        :param start_date: start time of ensemble
        :param end_date: end time of ensemble
        :param file_deck: ATCF file deck; one of `a`, `b`, `f`
        :param advisories: ATCF advisory type; one of ``BEST``, ``OFCL``, ``OFCP``, ``HMON``, ``CARQ``, ``HWRF``
        """

        track = VortexTrack.from_file(
            filename,
            start_date=start_date,
            end_date=end_date,
            file_deck=file_deck,
            advisories=advisories,
        )
        if file_deck in ['a', ATCF_FileDeck('a')]:
            track.forecast_time = (
                track.data.track_start_time.min() if start_date is None else start_date
            )
        instance = cls.from_track(track, features)
        instance.__filename = filename
        return instance

    @classmethod
    def from_track(
        cls,
        track: VortexTrack,
        features: Optional[PerturberFeatures] = PerturberFeatures.ISOTACH_ADJUSTMENT,
    ) -> 'VortexPerturber':
        """
        build storm perturber from an existing `VortexTrack` object

        :param track: `VortexTrack` object
        """

        start_date = track.start_date
        if track.forecast_time is not None:
            start_date = track.forecast_time
        instance = cls(
            track.nhc_code,
            start_date=start_date,
            end_date=track.end_date,
            file_deck=track.file_deck,
            advisories=track.advisories,
            features=features,
        )
        instance.track = track
        instance.__previous_configuration = {
            'storm': track.nhc_code,
            'start_date': start_date,
            'end_date': track.end_date,
            'file_deck': track.file_deck,
            'advisories': track.advisories,
        }
        return instance

    @property
    def start_date(self) -> datetime:
        return self.__start_date

    @start_date.setter
    def start_date(self, start_date: datetime):
        if start_date is not None and not isinstance(start_date, datetime):
            start_date = parse_date(start_date)
        self.__start_date = start_date

    @property
    def end_date(self) -> datetime:
        return self.__end_date

    @end_date.setter
    def end_date(self, end_date: datetime):
        if end_date is not None and not isinstance(end_date, datetime):
            end_date = parse_date(end_date)
        self.__end_date = end_date

    @property
    def file_deck(self) -> ATCF_FileDeck:
        return self.__file_deck

    @file_deck.setter
    def file_deck(self, file_deck: ATCF_FileDeck):
        if file_deck is not None and not isinstance(file_deck, datetime):
            file_deck = typepigeon.convert_value(file_deck, ATCF_FileDeck)
        self.__file_deck = file_deck

    @property
    def track(self) -> VortexTrack:
        configuration = {
            'storm': self.storm,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'file_deck': self.file_deck,
            'advisories': self.advisories,
        }

        is_equal = False
        if self.__previous_configuration is not None:
            for key in configuration:
                if not isinstance(
                    configuration[key], type(self.__previous_configuration[key])
                ):
                    break
                elif isinstance(configuration[key], DataFrame) and isinstance(
                    self.__previous_configuration[key], DataFrame
                ):
                    if not configuration[key].equals(self.__previous_configuration[key]):
                        break
                elif configuration[key] != self.__previous_configuration[key]:
                    break
            else:
                is_equal = True

        if not is_equal:
            if self.__filename is not None and self.__filename.exists():
                self.__track = VortexTrack.from_file(
                    self.__filename,
                    file_deck=configuration['file_deck'],
                    advisories=configuration['advisories'],
                    start_date=configuration['start_date'],
                    end_date=configuration['end_date'],
                )
            else:
                self.__track = VortexTrack(**configuration)
            if configuration['file_deck'] in ['a', ATCF_FileDeck('a')]:
                self.__track.forecast_time = (
                    self.__track.data.track_start_time.min()
                    if configuration['start_date'] is None
                    else configuration['start_date']
                )
            self.__previous_configuration = configuration

        if self.__track.nhc_code is not None:
            self.storm = self.__track.nhc_code

        return self.__track

    @track.setter
    def track(self, track: VortexTrack):
        self.__track = track

    def write(
        self,
        perturbations: Union[int, List[float], List[Dict[str, float]], None],
        variables: List[VortexVariable],
        directory: PathLike = None,
        sample_from_distribution: bool = False,
        sample_rule: str = 'random',
        sample_division_fraction: float = 0.99,
        quadrature: bool = False,
        quadrature_order: int = 3,
        quadrature_rule: str = 'Gaussian',
        sparse_quadrature: bool = False,
        weights: List[float] = None,
        overwrite: bool = False,
        continue_numbering: bool = False,
        parallel: bool = True,
    ) -> List[Path]:
        """
        :param perturbations: either the number of perturbations to create, or a list of floats meant to represent points on either the standard Gaussian distribution or a bounded uniform distribution
        :param variables: list of variable names, any combination of `["max_sustained_wind_speed", "radius_of_maximum_winds", "along_track", "cross_track"]`
        :param directory: directory to which to write
        :param sample_from_distribution: override given perturbations with random samples from the joint distribution
        :param sample_rule: rule to use for the distribution sampling. Please choose from:
               ``random`` [default], ``sobol``, ``halton``, ``hammersley``, ``korobov``, ``additive_recursion``, or ``latin_hypercube``, ``equal_division``
        :param sample_division_fraction: the fraction of the distribution to cover for ``equal_division`` sampling option
        :param quadrature: add perturbations along quadrature
        :param quadrature_order: order of the quadrature
        :param quadrature_rule: rule of the quadrature for generating abscissas and weights
        :param sparse_quadrature: use Smolyak’s sparse grid instead of normal tensor product grid
        :param weights: weights to use with perturbations
        :param overwrite: overwrite existing files
        :param continue_numbering: continue the existing numbering scheme if files already exist in the output directory
        :param parallel: generate perturbations concurrently
        :returns: written filenames
        """

        for index, variable in enumerate(variables):
            if isinstance(variable, type):
                variables[index] = variable()
            elif isinstance(variable, str):
                for existing_variable in VortexPerturbedVariable.__subclasses__():
                    if variable == existing_variable.name:
                        variables[index] = existing_variable()
                        break

        if directory is None:
            directory = Path.cwd()
        elif not isinstance(directory, Path):
            directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

        variable_names = [variable.name for variable in variables]

        if isinstance(perturbations, int):
            num_perturbations = perturbations
            perturbations = numpy.full(
                (num_perturbations, len(variables)), fill_value=numpy.nan
            )
        elif perturbations is None:
            if not quadrature:
                raise ValueError('cannot infer number of perturbations from given information')
            num_perturbations = None
        else:
            if sample_from_distribution:
                LOGGER.warning(
                    'overwriting given perturbations with random samples from joint distribution'
                )
            num_perturbations = len(perturbations)
            perturbations_array = []
            for perturbation in perturbations:
                if isinstance(perturbation, float):
                    perturbation = numpy.full((len(variables),), fill_value=perturbation)
                elif isinstance(perturbation, Mapping):
                    perturbation_array = []
                    for variable in variables:
                        if variable in perturbation:
                            variable_perturbation = perturbation[variable]
                        elif variable.__class__ in perturbation:
                            variable_perturbation = perturbation[variable.__class__]
                        elif variable.name in perturbation:
                            variable_perturbation = perturbation[variable.name]
                        else:
                            variable_perturbation = numpy.nan
                        perturbation_array.append(variable_perturbation)
                    perturbation = perturbation_array
                else:
                    perturbation = numpy.full((len(variables),), fill_value=numpy.nan)
                perturbations_array.append(perturbation)
            perturbations = numpy.array(perturbations_array)

        if sample_from_distribution:
            filetype_string = f'{sample_rule}'
        else:
            filetype_string = 'perturbation'
        if continue_numbering:
            # TODO: figure out how to continue perturbations by-variable (i.e. keep track of multiple series with different variables but the same total number of variables)
            existing_filenames = glob(
                str(directory / f'vortex_{len(variables)}_variable_{filetype_string}_*.22')
            )
            if len(existing_filenames) > 0:
                existing_filenames = sorted(existing_filenames)
                last_index = int(existing_filenames[-1][-4])
            else:
                last_index = 0
        else:
            last_index = 0

        run_names = [
            f'vortex_{len(variables)}_variable_{filetype_string}_{index + 1}'
            for index in range(last_index, last_index + num_perturbations)
        ]

        perturbations = [
            xarray.DataArray(
                perturbations,
                coords={'run': run_names, 'variable': variable_names},
                dims=('run', 'variable'),
            )
        ]

        if weights is None:
            weights = numpy.full((len(run_names),), fill_value=1.0)
        weights = [xarray.DataArray(weights, coords={'run': run_names}, dims=('run',))]

        distribution = distribution_from_variables(variables)

        if sample_from_distribution:
            # overwrite given perturbations with random samples from joint distribution
            if sample_rule == 'equal_division':
                random_sample = equal_division_sample(
                    distribution, num_perturbations, edge=sample_division_fraction
                )
            else:
                random_sample = distribution.sample(num_perturbations, rule=sample_rule)
            if len(variables) == 1:
                random_sample = random_sample.reshape(-1, 1)
            else:
                random_sample = random_sample.T

            perturbations[0] = xarray.DataArray(
                random_sample,
                coords={'run': run_names, 'variable': variable_names},
                dims=('run', 'variable'),
            )

        if quadrature:
            quadrature_nodes, quadrature_weights = chaospy.generate_quadrature(
                order=quadrature_order,
                dist=distribution,
                rule=quadrature_rule,
                sparse=sparse_quadrature,
            )

            quadrature_run_names = [
                f'vortex_{quadrature_nodes.shape[0]}_variable_quadrature_{index + 1}'
                for index in range(quadrature_nodes.shape[1])
            ]

            quadrature_nodes = xarray.DataArray(
                quadrature_nodes,
                coords={'run': quadrature_run_names, 'variable': variable_names},
                dims=('variable', 'run'),
            )
            perturbations.append(quadrature_nodes.T)

            weights.append(
                xarray.DataArray(
                    quadrature_weights, coords={'run': quadrature_run_names}, dims=('run',)
                )
            )

        perturbations.append(
            xarray.DataArray(
                numpy.full((1, len(variables)), fill_value=numpy.nan),
                coords={'run': ['original'], 'variable': variable_names},
                dims=('run', 'variable'),
            )
        )
        weights.append(xarray.DataArray([1.0], coords={'run': ['original']}, dims=('run',)))

        perturbations = xarray.merge(
            [
                xarray.combine_nested(perturbations, concat_dim='run').to_dataset(
                    name='perturbations'
                ),
                xarray.combine_nested(weights, concat_dim='run').to_dataset(name='weights'),
            ]
        )

        LOGGER.info(f'writing {len(perturbations["run"])} perturbations')

        if parallel:
            process_pool = ProcessPoolExecutorStackTraced()
        else:
            process_pool = None

        # for each variable, perturb the values and write each to a new `fort.22`
        futures = []
        for run_name in perturbations['run'].values:
            output_filename = directory / (run_name + '.22')
            if not output_filename.exists() or overwrite:
                # setting the alpha to the value from the input list
                perturbation = perturbations.sel(run=run_name)

                perturbation_values = perturbation['perturbations']
                if isinstance(perturbation_values, xarray.DataArray):
                    perturbation_values = {
                        variable: float(perturbation_values.sel(variable=variable).values)
                        for variable in variable_names
                    }
                elif not isinstance(perturbation_values, Mapping):
                    perturbation_values = {
                        variable: perturbation_values for variable in variable_names
                    }

                perturbation_values = {
                    variable.name
                    if (
                        issubclass(variable, VortexVariable)
                        if isinstance(variable, type)
                        else isinstance(variable, VortexVariable)
                    )
                    else variable: alpha
                    for variable, alpha in perturbation_values.items()
                }

                write_kwargs = {
                    'filename': output_filename,
                    'perturbation': perturbation_values,
                    'variables': copy(variable_names),
                    'weight': float(perturbation['weights'].values),
                }

                if parallel:
                    futures.append(
                        process_pool.submit(self.write_perturbed_track, **write_kwargs)
                    )
                else:
                    self.write_perturbed_track(**write_kwargs)

        if len(futures) > 0:
            output_filenames = [
                completed_future.result()
                for completed_future in concurrent.futures.as_completed(futures)
            ]
        else:
            output_filenames = [
                directory / (run_name + '.22') for run_name in perturbations['run'].values
            ]

        return output_filenames

    def write_perturbed_track(
        self,
        filename: PathLike,
        perturbation: Dict[str, float],
        variables: List[VortexPerturbedVariable],
        weight: float = None,
    ) -> Path:
        if not isinstance(filename, Path):
            filename = Path(filename)

        dataframe = self.track.data.copy()

        variable_names = {
            **{
                subclass.__name__: subclass
                for subclass in VortexPerturbedVariable.__subclasses__()
            },
            **{
                subclass.name: subclass
                for subclass in VortexPerturbedVariable.__subclasses__()
            },
        }
        for variable_index, variable in enumerate(variables):
            if isinstance(variable, str):
                variables[variable_index] = variable_names[variable]()

        # add units to data frame
        dataframe = dataframe.astype(
            {
                variable.name: PintType(variable.unit)
                for variable in variables
                if variable.name in dataframe
            },
            copy=False,
        )

        for index, variable in enumerate(variables):
            if isinstance(variable, str):
                for variable_class in VortexPerturbedVariable.__subclasses__():
                    if (
                        variable.lower() == variable_class.name.lower()
                        or variable.lower() == variable_class.__name__.lower()
                    ):
                        variables[index] = variable_class()

        # add units to data frame
        dataframe = dataframe.astype(
            {
                variable.name: PintType(variable.unit)
                for variable in variables
                if variable.name in dataframe
            },
            copy=False,
        )

        for variable in variables:
            if variable.name in perturbation:
                alpha = perturbation[variable.name]
            else:
                alpha = 0

            if alpha is None or abs(alpha) > 1.0e-3:
                # Make the random pertubations based on the historical forecast errors
                # Interpolate from the given VT to the storm_VT

                # Get the historical forecasting errors from initial storm state (intensity or size)
                historical_forecast_errors = variable.storm_errors(self.track.data)

                validation_hours = self.validation_times / timedelta(hours=1)

                if self.file_deck.value == 'b':
                    # hindcast errors based on the 0-hr historical averages
                    data_dict = {}
                    for column in historical_forecast_errors.columns:
                        column_error = []
                        for rdx, row in enumerate(validation_hours):
                            column_error.append(
                                variable.storm_errors(self.track.data, rdx)
                                .loc[0, column]
                                .magnitude
                            )
                        data_dict[column] = column_error
                    validation_time_errors = DataFrame(data=data_dict, index=validation_hours,)
                else:
                    # forecast errors than grow with time based on initial intensity
                    validation_time_errors = DataFrame(
                        data={
                            column: interp(
                                x=validation_hours,
                                xp=historical_forecast_errors.index,
                                fp=historical_forecast_errors.loc[
                                    :, column
                                ].values.quantity.magnitude,
                            )
                            for column in historical_forecast_errors.columns
                        },
                        index=validation_hours,
                    )

                # get the random perturbation sample
                if variable.perturbation_type == PerturbationType.GAUSSIAN:
                    if alpha is None:
                        alpha = gauss(0, 1)
                        perturbation[variable.name] = alpha

                    LOGGER.debug(f'gaussian alpha = {alpha}')
                    perturbed_values = (
                        validation_time_errors.iloc[:, 0] * alpha / 0.7979
                    ).values
                    if (
                        variable.unit is not None
                        and variable.unit != variable.unit._REGISTRY.dimensionless
                    ):
                        perturbed_values *= variable.unit

                    # add the error to the variable with bounds to some physical constraints
                    dataframe = variable.perturb(
                        dataframe,
                        values=perturbed_values,
                        times=self.validation_times,
                        inplace=True,
                    )
                elif variable.perturbation_type == PerturbationType.UNIFORM:
                    if alpha is None:
                        alpha = uniform(-1, 1)
                        perturbation[variable.name] = alpha

                    LOGGER.debug(f'uniform alpha in [-1,1] = {alpha}')
                    # extrapolate to 0th/100th percentile...
                    min_validation_time_errors = (
                        1.3 * validation_time_errors.loc[:, '15th percentile error']
                        - 0.3 * validation_time_errors.loc[:, '50th percentile error']
                    )
                    max_validation_time_errors = (
                        1.3 * validation_time_errors.loc[:, '85th percentile error']
                        - 0.3 * validation_time_errors.loc[:, '50th percentile error']
                    )
                    ## if just choose 15th/85th percentile...
                    # min_validation_time_errors = validation_time_errors.loc[:, '15th percentile error']
                    # max_validation_time_errors = validation_time_errors.loc[:, '85th percentile error']
                    perturbed_values = (
                        0.5
                        * (
                            min_validation_time_errors * (1 - alpha)
                            + max_validation_time_errors * (1 + alpha)
                        ).values
                    )
                    if variable.unit is not None and variable.unit != units.dimensionless:
                        perturbed_values *= variable.unit

                    # subtract the error from the variable with physical constraint bounds
                    dataframe = variable.perturb(
                        dataframe,
                        values=perturbed_values,
                        times=self.validation_times,
                        inplace=True,
                    )
                else:
                    raise NotImplementedError(
                        f'perturbation type "{variable.perturbation_type}" is not recognized'
                    )

        # remove units from data frame
        for column in dataframe:
            if isinstance(dataframe[column].dtype, PintType):
                dataframe[column] = dataframe[column].pint.magnitude

        # enter if it is not the original unperturbed file
        if any(~numpy.isnan(list(perturbation.values()))):
            # Compute potential changes in the central pressure in accordance with Holland B relationship
            dataframe[CentralPressure.name] = self.compute_pc_from_Vmax(dataframe)

            if self.__features & PerturberFeatures.ISOTACH_ADJUSTMENT:
                # Compute potential changes to r34/50,64 radii at all quadrants in accordance with the GAHM profile
                quadrants = [
                    IsotachRadius(IsotachRadiusQuadrant.NEQ),
                    IsotachRadius(IsotachRadiusQuadrant.SEQ),
                    IsotachRadius(IsotachRadiusQuadrant.SWQ),
                    IsotachRadius(IsotachRadiusQuadrant.NWQ),
                ]
                for quadrant in quadrants:
                    dataframe = quadrant.adjust(
                        vortex_dataframe=dataframe,
                        original_dataframe=self.track.data,
                        inplace=True,
                    )
                # remove any rows that now have all 0 isotach radii for the 50-kt or 64-kt isotach
                quadrant_names = [q.column for q in quadrants]
                drop_row = (dataframe[quadrant_names] == 0).all(axis=1) & (
                    dataframe['isotach_radius'].isin([50, 64])
                )
                dataframe.drop(index=dataframe.index[drop_row], inplace=True)

        # write out the modified `fort.22`
        VortexTrack(storm=dataframe).to_file(filename, overwrite=True)

        if weight is not None:
            perturbation['weight'] = weight
        with open(filename.parent / f'{filename.stem}.json', 'w') as output_json:
            json.dump(perturbation, output_json, indent=2)

        LOGGER.info(
            f'wrote {len(variables)}-variable perturbation to "{os.path.relpath(filename, Path.cwd())}"'
        )

        return filename

    @property
    def validation_times(self) -> List[timedelta]:
        """ get the validation time of storm """
        return self.track.data['datetime'] - self.track.start_date

    @property
    def holland_B(self) -> Quantity:
        """ Compute Holland B at each time snap """
        dataframe = self.track.data
        # get Vmax after subtracting speed and adjusting to boundary layer
        Vmax, _ = MaximumSustainedWindSpeed().radial_Vmax_at_BL(dataframe)
        DelP = dataframe[BackgroundPressure.name] - dataframe[CentralPressure.name]
        DelP = DelP.values * CentralPressure.unit
        B = Vmax * Vmax * AIR_DENSITY * E1 / DelP
        return B

    def compute_pc_from_Vmax(self, dataframe: DataFrame) -> float:
        """ Compute central pressure from Vmax based on Holland B """
        # get Vmax after subtracting speed and adjusting to boundary layer height
        Vmax, _ = MaximumSustainedWindSpeed().radial_Vmax_at_BL(dataframe)
        DelP = Vmax * Vmax * AIR_DENSITY * E1 / self.holland_B
        pc = dataframe[BackgroundPressure.name].values * CentralPressure.unit - DelP
        return pc.magnitude


def distribution_from_variables(variables: List[VortexPerturbedVariable]) -> Distribution:
    """
    :param variables: names of perturbed variables
    :return: chaospy joint distribution encompassing variables
    """

    if variables is None or len(variables) == 0:
        variables = [variable() for variable in VortexPerturbedVariable.__subclasses__()]

    return chaospy.J(*(variable.chaospy_distribution() for variable in variables))


def equal_division_sample(
    distribution: Distribution, num_perturbations: int, edge: float = 0.99
):
    """
    :param variables: names of perturbed variables
    :param num_perturbations: number of samples to retrieve
    :return: samples
    """

    # get the edge percentile uniform distribution
    sample_uniform = numpy.linspace(-edge, +edge, num_perturbations, endpoint=True)
    # Transform the uniform dimension into gaussian
    sample_gaussian = erfinv(sample_uniform) * sqrt(2)

    # use ordering from korobov samples
    korobov_samples = chaospy.create_korobov_samples(num_perturbations, len(distribution))
    samples_order = numpy.argsort(korobov_samples)

    # add the magnitudes here
    samples = numpy.empty(samples_order.shape)
    for dx, dist in enumerate(distribution):
        if isinstance(dist, chaospy.Normal):
            samples[dx, :] = sample_gaussian[samples_order[dx, :]]
        elif isinstance(dist, chaospy.Uniform):
            samples[dx, :] = sample_uniform[samples_order[dx, :]]
        else:
            raise f'distribution {dist} not implemented'

    return samples


def utm_crs_from_longitude(longitude: float) -> CRS:
    """
    utm_from_lon - UTM zone for a longitude
    Not right for some polar regions (Norway, Svalbard, Antarctica)

    :param longitude: longitude
    :return: coordinate reference system
    """

    return CRS.from_epsg(32600 + int(floor((longitude + 180) / 6) + 1))


def get_offset(
    point_1: (float, float), point_2: (float, float), distance: float
) -> (float, float):
    """
    get the perpendicular offset to the line (x1,y1) -> (x2,y2) by the specified distance

    :param point_1: first point
    :param point_2: second point
    :param distance: distance
    :returns: offset
    """

    unit = None
    for value in (point_1, point_2, distance):
        if isinstance(value, Quantity):
            unit = value._REGISTRY.meter
            break

    if numpy.all(point_1[:2] == point_2[:2]):
        offset = [0, 0]
    elif point_1[0] == point_2[0]:
        if isinstance(distance, Quantity):
            distance = distance.to(unit).magnitude
        offset = [distance, 0]
    elif point_1[1] == point_2[1]:
        if isinstance(distance, Quantity):
            distance = distance.to(unit).magnitude
        offset = [0, distance]
    else:
        # if z is the distance to your parallel curve,
        # then your delta-x and delta-y calculations are:
        #   z**2 = x**2 + y**2
        #   y = pslope * x
        #   z**2 = x**2 + (pslope * x)**2
        #   z**2 = x**2 + pslope**2 * x**2
        #   z**2 = (1 + pslope**2) * x**2
        #   z**2 / (1 + pslope**2) = x**2
        #   z / (1 + pslope**2)**0.5 = x

        points = numpy.vstack([point_1, point_2])

        difference = numpy.diff(points, axis=0).ravel()

        # tangential slope approximation
        slope = difference[1] / difference[0]

        # normal slope
        normal_slope = -1 / slope  # (might be 1/slope depending on direction of travel)
        normal_sign = ((normal_slope > 0) == (difference[0] < 0)) * 2 - 1

        dx = normal_sign * distance / sqrt(1 + normal_slope ** 2)
        dy = normal_slope * dx

        if isinstance(dx, Quantity):
            dx = dx.to(unit).magnitude
        if isinstance(dy, Quantity):
            dy = dy.to(unit).magnitude

        offset = [dx, dy]

    offset = numpy.array(offset)

    if unit is not None:
        offset *= unit

    return offset


def parse_vortex_perturbations(directory: PathLike = None) -> Dataset:
    """
    parse `fort.22` and JSON files into a xarray dataset

    :param directory: directory containing `fort.22` and JSON files of tracks
    :returns: array of variables perturbed with each track
    """

    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)

    # reading the JSON data using json.load()
    perturbations = {}
    for filename in directory.glob('**/vortex*.json'):
        with open(filename) as vortex_file:
            perturbations[filename.stem] = json.load(vortex_file)

    if len(perturbations) == 0:
        raise FileNotFoundError(
            f'could not find any perturbation JSON file(s) in "{directory}"'
        )

    run_names = sorted(perturbations, key=lambda run_name: int(run_name.split('_')[-1]))
    variable_names = [
        variable_name
        for variable_name in perturbations[run_names[0]]
        if variable_name != 'weight'
    ]

    perturbation_values = []
    weights = []
    for run_name in run_names:
        run_perturbations = perturbations[run_name]
        perturbation_values.append(
            [
                run_perturbations[variable_name]
                if variable_name in run_perturbations
                else numpy.nan
                for variable_name in variable_names
            ]
        )
        weights.append(
            run_perturbations['weight'] if 'weight' in run_perturbations else numpy.nan
        )

    return Dataset(
        {
            'perturbations': (('run', 'variable'), perturbation_values),
            'weights': (('run',), weights),
        },
        coords={'run': run_names, 'variable': variable_names},
    )


def perturb_tracks(
    perturbations: Union[int, List[float], List[Dict[str, float]]],
    directory: PathLike = None,
    storm: Union[str, PathLike] = None,
    variables: List[VortexPerturbedVariable] = None,
    sample_from_distribution: bool = False,
    sample_rule: str = 'random',
    quadrature: bool = False,
    quadrature_order: int = 3,
    quadrature_rule: str = 'Gaussian',
    sparse_quadrature: bool = False,
    start_date: datetime = None,
    end_date: datetime = None,
    file_deck: ATCF_FileDeck = None,
    advisories: List[str] = None,
    overwrite: bool = False,
    parallel: bool = True,
    features: Optional[PerturberFeatures] = PerturberFeatures.ISOTACH_ADJUSTMENT,
):
    """
    write a set of perturbed storm tracks

    :param perturbations: either the number of perturbations to create, or a list of floats meant to represent points on either the standard Gaussian distribution or a bounded uniform distribution
    :param directory: directory to which to write
    :param storm: ATCF storm ID, or file path to an existing `fort.22` / ATCF file, from which to perturb
    :param variables: vortex variables to perturb
    :param sample_from_distribution: override given perturbations with random samples from the joint distribution
    :param sample_rule: rule to use for the distribution sampling. Please choose from:
           ``random`` [default], ``sobol``, ``halton``,``hammersley``, ``korobov``, ``additive_recursion``, or ``latin_hypercube``
    :param quadrature: add perturbations along quadrature
    :param quadrature_order: order of the quadrature
    :param quadrature_rule: rule of the quadrature for generating abscissas and weights
    :param sparse_quadrature: use Smolyak’s sparse grid instead of normal tensor product grid
    :param start_date: model start time of ensemble
    :param end_date: model end time of ensemble
    :param file_deck: letter of file deck, one of `a`, `b`
    :param advisories: record types (i.e. `BEST`, `OFCL`)
    :param overwrite: overwrite existing files
    :param parallel: generate perturbations concurrently
    :return: mapping of track names to perturbation JSONs
    """

    if directory is None:
        directory = Path.cwd()
    elif not isinstance(directory, Path):
        directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    if storm is None:
        storm = directory / 'original.22'

    try:
        if Path(storm).exists():
            perturber = VortexPerturber.from_file(
                storm,
                start_date=start_date,
                end_date=end_date,
                file_deck=file_deck,
                advisories=advisories,
                features=features,
            )
        else:
            raise FileNotFoundError
    except:
        if storm is None:
            raise ValueError('no storm ID specified')

        perturber = VortexPerturber(
            storm=storm,
            start_date=start_date,
            end_date=end_date,
            file_deck=file_deck,
            advisories=advisories,
            features=features,
        )

    filenames = [directory / 'original.22']
    filenames += perturber.write(
        perturbations=perturbations,
        variables=variables,
        directory=directory,
        sample_from_distribution=sample_from_distribution,
        sample_rule=sample_rule,
        quadrature=quadrature,
        quadrature_order=quadrature_order,
        quadrature_rule=quadrature_rule,
        sparse_quadrature=sparse_quadrature,
        overwrite=overwrite,
        parallel=parallel,
    )

    perturbations = {
        track_filename.stem: {
            'besttrack': {
                'fort22_filename': Path(os.path.relpath(track_filename, directory.parent))
            }
        }
        for index, track_filename in enumerate(filenames)
    }

    return perturbations
