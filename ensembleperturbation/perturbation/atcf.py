#! /usr/bin/env python3
"""
Script to:
(1) extract an ATCF best track dataset;
(2) randomly perturb different parameters (e.g., intensity,
size, track coordinates) to generate an ensemble; and
(3) write out each ensemble member to the fort.22 ATCF
tropical cyclone vortex format file.

Variables that can be perturbed:
- "max_sustained_wind_speed" (Vmax) is made weaker/stronger
  based on random gaussian distribution with sigma scaled by
  historical mean absolute errors. central_pressure (pc)
  is then changed proportionally based on Holland B

- "radius_of_maximum_winds" (Rmax) is made small/larger
  based on random number in a range bounded by the 15th and
  85th percentile CDF of historical forecast errors.

- "along_track" variable is used to offset the coordinate of
  the tropical cyclone center at each forecast time forward or
  backward along the given track based on a random gaussian
  distribution with sigma scaled by historical mean absolute
  errors.

- "cross_track" variable is used to offset the coordinate of
  the tropical cyclone center at each forecast time a certain
  perpendicular distance from the given track based on a
  random gaussian distribution with sigma scaled by historical
  mean absolute errors.

By William Pringle, Argonne National Laboratory, Mar-May 2021
   Zach Burnett, NOS/NOAA
   Saeed Moghimi, NOS/NOAA
"""

from abc import ABC
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from glob import glob
import json
from math import exp, inf, sqrt
import os
from os import PathLike
from pathlib import Path
from random import gauss, uniform
from tempfile import TemporaryDirectory
from typing import Dict, List, Mapping, Union
import warnings

from adcircpy.forcing.winds.best_track import convert_value, FileDeck, Mode, VortexForcing
import chaospy
from dateutil.parser import parse as parse_date
import numpy
from numpy import floor, interp, sign
import pandas
from pandas import DataFrame
from pandas.core.common import SettingWithCopyWarning
import pint
from pint import Quantity, UnitStrippedWarning
from pint_pandas import PintArray, PintType
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection
from shapely.geometry import LineString

from ensembleperturbation.utilities import get_logger, units

LOGGER = get_logger('perturbation.atcf')

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UnitStrippedWarning)

AIR_DENSITY = 1.15 * units.kilogram / units.meters ** 3

E1 = exp(1.0)  # e

# Index of absolute errors (forecast times [hrs)]
HISTORICAL_ERROR_HOURS = [0, 12, 24, 36, 48, 60, 72, 96, 120]  # has 60-hr data (for Rmax)
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
        historical_forecast_errors: {str: DataFrame} = None,
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
    def historical_forecast_errors(self) -> {str: DataFrame}:
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
                        dataframe[column].pint.ito(self.unit)
                    dataframe[column].astype(pint_type, copy=False)
        return self.__historical_forecast_errors

    @historical_forecast_errors.setter
    def historical_forecast_errors(self, historical_forecast_errors: {str: DataFrame}):
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
                        dataframe[column].pint.ito(self.unit)
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
        values: [float],
        times: [datetime],
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

        variable_values = vortex_dataframe[self.name].values
        if (
            not isinstance(variable_values, PintArray)
            or variable_values.units == variable_values.units._REGISTRY.dimensionless
        ):
            variable_values *= self.unit
        if (
            not isinstance(values, Quantity)
            or values.units == values.units._REGISTRY.dimensionless
        ):
            values *= self.unit

        all_values = variable_values - values
        vortex_dataframe[self.name] = [
            min(self.upper_bound, max(value, self.lower_bound)).magnitude
            for value in all_values
        ] * self.unit

        return vortex_dataframe


class MaximumSustainedWindSpeed(VortexPerturbedVariable):
    name = 'max_sustained_wind_speed'
    perturbation_type = PerturbationType.GAUSSIAN

    # Reference - 2019_Psurge_Error_Update_FINAL.docx
    # Table 12: Adjusted intensity errors [kt] for 2015-2019
    def __init__(self):
        super().__init__(
            lower_bound=25,
            upper_bound=165,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {'mean error [kt]': [1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91]},
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {'mean error [kt]': [2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62]},
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
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
            },
            unit=units.knot,
        )


class RadiusOfMaximumWinds(VortexPerturbedVariable):
    name = 'radius_of_maximum_winds'
    perturbation_type = PerturbationType.UNIFORM

    def __init__(self):
        super().__init__(
            lower_bound=5,
            upper_bound=200,
            historical_forecast_errors={
                '<15sm': DataFrame(
                    {
                        'minimum error [sm]': [
                            0.0,
                            -13.82,
                            -19.67,
                            -21.37,
                            -26.31,
                            -32.71,
                            -39.12,
                            -46.80,
                            -52.68,
                        ],
                        'maximum error [sm]': [
                            0.0,
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
                        'minimum error [sm]': [
                            0.0,
                            -10.47,
                            -14.54,
                            -20.35,
                            -23.88,
                            -21.78,
                            -19.68,
                            -24.24,
                            -28.30,
                        ],
                        'maximum error [sm]': [
                            0.0,
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
                        'minimum error [sm]': [
                            0.0,
                            -8.57,
                            -13.41,
                            -10.87,
                            -9.26,
                            -9.34,
                            -9.42,
                            -7.41,
                            -7.40,
                        ],
                        'maximum error [sm]': [
                            0.0,
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
                        'minimum error [sm]': [
                            0.0,
                            -10.66,
                            -7.64,
                            -5.68,
                            -3.25,
                            -1.72,
                            -0.19,
                            3.65,
                            2.59,
                        ],
                        'maximum error [sm]': [
                            0.0,
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
                        'minimum error [sm]': [
                            0.0,
                            -15.36,
                            -10.37,
                            3.14,
                            12.10,
                            12.21,
                            12.33,
                            6.66,
                            7.19,
                        ],
                        'maximum error [sm]': [
                            0.0,
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


class CrossTrack(VortexPerturbedVariable):
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
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
            },
            unit=units.nautical_mile,
        )

    def perturb(
        self,
        vortex_dataframe: DataFrame,
        values: [float],
        times: [datetime],
        inplace: bool = False,
    ) -> DataFrame:
        """
        offset_track(df_,VT,cross_track_errors)
          - Offsets points by a given perpendicular error/distance from the original track

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
            normal_offset = numpy.mean([previous_offset, next_offset])
            alpha = abs(cross_track_error) / numpy.sqrt(numpy.sum(normal_offset ** 2))

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
                    index=HISTORICAL_ERROR_HOURS_NO_60H,
                ),
            },
            unit=units.nautical_mile,
        )

    def perturb(
        self,
        vortex_dataframe: DataFrame,
        values: [float],
        times: [datetime],
        inplace: bool = False,
    ) -> DataFrame:
        """
        interpolate_along_track(df_,VT,along_track_errros)
        Offsets points by a given error/distance by interpolating along the track

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
        hours = numpy.concatenate((hours[0] - previous_diffs, hours, hours[-1] + after_diffs))

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
            projected_coordinate = line_segment.interpolate(abs(along_error.magnitude))

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


class VortexPerturber:
    def __init__(
        self,
        storm: str,
        start_date: datetime = None,
        end_date: datetime = None,
        file_deck: FileDeck = None,
        mode: Mode = None,
        record_type: str = None,
    ):
        """
        build storm perturber

        :param storm: NHC storm code, for instance `al062018`
        :param start_date: start time of ensemble
        :param end_date: end time of ensemble
        :param file_deck: letter of file deck, one of `a`, `b`
        :param mode: either `realtime` / `aid_public` or `historical` / `archive`
        :param record_type: record type (i.e. `BEST`, `OFCL`)
        """

        self.__storm = None
        self.__start_date = None
        self.__end_date = None
        self.__file_deck = None
        self.__mode = None
        self.__forcing = None
        self.__previous_configuration = None

        self.storm = storm
        self.start_date = start_date
        self.end_date = end_date
        self.file_deck = file_deck
        self.mode = mode
        self.record_type = record_type

        self.__filename = None

    @property
    def storm(self) -> str:
        return self.__storm

    @storm.setter
    def storm(self, storm: str):
        self.__storm = storm

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
    def file_deck(self) -> FileDeck:
        return self.__file_deck

    @file_deck.setter
    def file_deck(self, file_deck: FileDeck):
        if file_deck is not None and not isinstance(file_deck, datetime):
            file_deck = convert_value(file_deck, FileDeck)
        self.__file_deck = file_deck

    @property
    def mode(self) -> Mode:
        return self.__mode

    @mode.setter
    def mode(self, mode: Mode):
        if mode is not None and not isinstance(mode, datetime):
            mode = convert_value(mode, Mode)
        self.__mode = mode

    @property
    def forcing(self) -> VortexForcing:
        configuration = {
            'storm': self.storm,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'file_deck': self.file_deck,
            'mode': self.mode,
            'record_type': self.record_type,
        }

        is_equal = False
        if self.__previous_configuration is not None:
            for key in configuration:
                if type(configuration[key]) is not type(self.__previous_configuration[key]):
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
            if self.__filename is not None:
                if '.22' in self.__filename.suffix:
                    self.__forcing = VortexForcing.from_fort22(
                        self.__filename,
                        start_date=configuration['start_date'],
                        end_date=configuration['end_date'],
                    )
                else:
                    self.__forcing = VortexForcing.from_atcf_file(
                        self.__filename,
                        start_date=configuration['start_date'],
                        end_date=configuration['end_date'],
                    )
            else:
                self.__forcing = VortexForcing(**configuration)
            self.__previous_configuration = configuration

        if self.__forcing.storm_id is not None:
            self.__storm = self.__forcing.storm_id

        return self.__forcing

    def write(
        self,
        perturbations: Union[int, List[float], List[Dict[str, float]]],
        variables: [VortexVariable],
        directory: PathLike = None,
        quadrature: bool = True,
        overwrite: bool = False,
        continue_numbering: bool = False,
        parallel: bool = True,
    ) -> [Path]:
        """
        :param perturbations: either the number of perturbations to create, or a list of floats meant to represent points on either the standard Gaussian distribution or a bounded uniform distribution
        :param variables: list of variable names, any combination of `["max_sustained_wind_speed", "radius_of_maximum_winds", "along_track", "cross_track"]`
        :param directory: directory to which to write
        :param quadrature: create
        :param overwrite: overwrite existing files
        :param continue_numbering: continue the existing numbering scheme if files already exist in the output directory
        :param parallel: generate perturbations concurrently
        :returns: written filenames
        """

        if isinstance(perturbations, int):
            perturbations = [None] * perturbations

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

        if quadrature:
            perturbations, weights = quadrature_perturbations(
                variables=variables, maximum=len(perturbations),
            )
        else:
            weights = None

        # write out original fort.22
        original_filename = directory / 'original.22'
        self.forcing.write(original_filename, overwrite=True)
        if self.__filename is None:
            self.__filename = original_filename

        # Get the initial intensity and size
        storm_strength = storm_intensity_class(
            self.compute_initial(MaximumSustainedWindSpeed.name),
        )
        storm_size = storm_size_class(self.compute_initial(RadiusOfMaximumWinds.name))

        print(f'Initial storm strength: {storm_strength}')
        print(f'Initial storm size: {storm_size}')

        # extracting original dataframe
        original_data = self.forcing.data

        LOGGER.info(f'writing {len(perturbations)} perturbations')

        if continue_numbering:
            # TODO: figure out how to continue perturbations by-variable (i.e. keep track of multiple series with different variables but the same total number of variables)
            existing_filenames = glob(str(directory / f'vortex_*_variable_perturbation_*.22'))
            if len(existing_filenames) > 0:
                existing_filenames = sorted(existing_filenames)
                last_index = int(existing_filenames[-1][-4])
            else:
                last_index = 0
        else:
            last_index = 0

        # for each variable, perturb the values and write each to a new `fort.22`
        output_filenames = [
            directory
            / f'vortex_{len(variables)}_variable_perturbation_{perturbation_number}.22'
            for perturbation_number in range(
                last_index + 1, len(perturbations) + last_index + 1
            )
        ]

        if parallel:
            process_pool = ProcessPoolExecutor()
            temporary_directory = TemporaryDirectory()
            original_data_pickle_filename = Path(temporary_directory.name) / 'original_data.df'
            original_data.to_pickle(original_data_pickle_filename)
        else:
            process_pool = None
            temporary_directory = None
            original_data_pickle_filename = None

        futures = []

        variables = [variable.name for variable in variables]
        for perturbation_index, output_filename in enumerate(output_filenames):
            if not output_filename.exists() or overwrite:
                # setting the alpha to the value from the input list
                perturbation = perturbations[perturbation_index]

                if perturbation is None:
                    perturbation = {variable: None for variable in variables}
                elif not isinstance(perturbation, Mapping):
                    perturbation = {variable: perturbation for variable in variables}

                perturbation = {
                    variable.name if isinstance(variable, VortexVariable) else variable: alpha
                    for variable, alpha in perturbation.items()
                }

                write_kwargs = {
                    'filename': output_filename,
                    'dataframe': original_data,
                    'perturbation': perturbation,
                    'variables': variables,
                    'storm_size': storm_size,
                    'storm_strength': storm_strength,
                }

                if weights is not None:
                    write_kwargs['weight'] = weights[perturbation_index]

                if parallel:
                    write_kwargs['dataframe'] = original_data_pickle_filename

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

        if temporary_directory is not None:
            temporary_directory.cleanup()

        return output_filenames

    def write_perturbed_track(
        self,
        filename: PathLike,
        dataframe: DataFrame,
        perturbation: {str: float},
        variables: [VortexPerturbedVariable],
        storm_size: str,
        storm_strength: str,
        weight: float = None,
    ) -> Path:
        if not isinstance(filename, Path):
            filename = Path(filename)

        if isinstance(dataframe, DataFrame):
            dataframe = dataframe.copy(deep=True)
        else:
            dataframe = pandas.read_pickle(dataframe)

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
            alpha = perturbation[variable.name]

            # Make the random pertubations based on the historical forecast errors
            # Interpolate from the given VT to the storm_VT
            if isinstance(variable, RadiusOfMaximumWinds):
                storm_classification = storm_size
            else:
                storm_classification = storm_strength

            validation_hours = self.validation_times / timedelta(hours=1)

            historical_forecast_errors = variable.historical_forecast_errors[
                storm_classification
            ]
            try:
                # need to dequantify dataframe from pint units to run `interp`, then requantify resulting dataframe
                historical_forecast_errors = historical_forecast_errors.pint.dequantify()
            except:
                pass
            validation_time_errors = DataFrame(
                data={
                    column: interp(
                        x=validation_hours,
                        xp=historical_forecast_errors.index,
                        fp=historical_forecast_errors.loc[:, column],
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
                perturbed_values = (validation_time_errors.iloc[:, 0] * alpha / 0.7979).values
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
                perturbed_values = numpy.mean(
                    numpy.stack(
                        (
                            validation_time_errors.iloc[:, 0] * (1 - alpha),
                            validation_time_errors.iloc[:, 1] * (1 + alpha),
                        ),
                        axis=1,
                    ),
                    axis=1,
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

            if isinstance(variable, MaximumSustainedWindSpeed):
                # In case of Vmax need to change the central pressure incongruence with it (obeying Holland B relationship)
                dataframe[CentralPressure.name] = self.compute_pc_from_Vmax(dataframe)

        # remove units from data frame
        for column in dataframe:
            if isinstance(dataframe[column].dtype, PintType):
                dataframe[column] = dataframe[column].pint.magnitude

        # write out the modified `fort.22`
        VortexForcing(storm=dataframe).write(filename, overwrite=True)

        if weight is not None:
            perturbation['weight'] = weight
        with open(filename.parent / f'{filename.stem}.json', 'w') as output_json:
            json.dump(perturbation, output_json, indent=2)

        LOGGER.info(
            f'wrote {len(variables)}-variable perturbation to "{os.path.relpath(filename, Path.cwd())}"'
        )

        return filename

    @property
    def validation_times(self) -> [timedelta]:
        """ get the validation time of storm """
        return self.forcing.datetime - self.forcing.start_date

    def compute_initial(self, var: str) -> float:
        """ the initial value of the input variable var (Vmax or Rmax) """
        return self.forcing.data[var].iloc[0]

    @property
    def holland_B(self) -> float:
        """ Compute Holland B at each time snap """
        dataframe = self.forcing.data
        Vmax = dataframe[MaximumSustainedWindSpeed.name]
        DelP = dataframe[BackgroundPressure.name] - dataframe[CentralPressure.name]
        B = Vmax * Vmax * AIR_DENSITY * E1 / DelP
        return B

    def compute_pc_from_Vmax(self, dataframe: DataFrame) -> float:
        """ Compute central pressure from Vmax based on Holland B """
        Vmax = dataframe[MaximumSustainedWindSpeed.name]
        DelP = Vmax ** 2 * AIR_DENSITY * E1 / self.holland_B
        pc = dataframe[BackgroundPressure.name] - DelP
        return pc

    @classmethod
    def from_file(
        cls, filename: PathLike, start_date: datetime = None, end_date: datetime = None,
    ):
        """
        build storm perturber from an existing `fort.22` or ATCF file

        :param filename: file path to `fort.22` / ATCF file
        :param start_date: start time of ensemble
        :param end_date: end time of ensemble
        """

        if not isinstance(filename, Path):
            filename = Path(filename)

        if filename.suffix == '.22':
            vortex = VortexForcing.from_fort22(
                filename, start_date=start_date, end_date=end_date
            )
        else:
            vortex = VortexForcing.from_atcf_file(
                filename, start_date=start_date, end_date=end_date
            )

        instance = cls(vortex.dataframe, start_date=start_date, end_date=end_date)
        instance.__filename = filename
        return instance


def storm_intensity_class(max_sustained_wind_speed: float) -> str:
    """
    Category for Vmax based intensity

    :param max_sustained_wind_speed: maximum sustained wind speed, in knots
    :return: intensity classification
    """

    if not isinstance(max_sustained_wind_speed, Quantity):
        max_sustained_wind_speed *= units.knot

    if max_sustained_wind_speed < 50 * units.knot:
        return '<50kt'  # weak
    elif max_sustained_wind_speed <= 95 * units.knot:
        return '50-95kt'  # medium
    else:
        return '>95kt'  # strong


def storm_size_class(radius_of_maximum_winds: float) -> str:
    """
    Category for Rmax based size

    :param radius_of_maximum_winds: radius of maximum winds, in nautical miles
    :return: size classification
    """

    if not isinstance(radius_of_maximum_winds, Quantity):
        radius_of_maximum_winds *= units.nautical_mile

    if radius_of_maximum_winds < 15 * units.us_statute_mile:
        return '<15sm'  # very small
    elif radius_of_maximum_winds < 25 * units.us_statute_mile:
        return '15-25sm'  # small
    elif radius_of_maximum_winds < 35 * units.us_statute_mile:
        return '25-35sm'  # medium
    elif radius_of_maximum_winds <= 45 * units.us_statute_mile:
        return '35-45sm'  # large
    else:
        return '>45sm'  # very large


def quadrature_perturbations(
    variables: [VortexPerturbedVariable], maximum: int = None
) -> (numpy.ndarray, numpy.ndarray):
    """
    Generate quadrature from variable distributions.

    :param variables: names of perturbed variables
    :returns: array of nodes with size NxV, array of weights with size N
    """

    if variables is None or len(variables) == 0:
        variables = [variable() for variable in VortexPerturbedVariable.__subclasses__()]

    distribution = chaospy.J(*(variable.chaospy_distribution() for variable in variables))

    nodes, weights = chaospy.generate_quadrature(
        order=3, dist=distribution, rule='Gaussian', n_max=maximum
    )

    perturbations = [
        {variable: node[index] for index, variable in enumerate(variables)} for node in nodes.T
    ]

    return perturbations, weights


def utm_crs_from_longitude(longitude: float) -> CRS:
    """
    utm_from_lon - UTM zone for a longitude
    Not right for some polar regions (Norway, Svalbard, Antartica)

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

        points = numpy.concatenate([point_1, point_2], axis=0)

        difference = numpy.diff(points, axis=0)

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


def parse_vortex_perturbations(
    directory: PathLike = None, output_filename: PathLike = None
) -> DataFrame:
    """
    Parse `fort.22` and JSON files into a dataframe.

    :param directory: directory containing `fort.22` and JSON files of tracks
    :returns: dataframe of the variables perturbed with each track
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

    # convert dictionary to dataframe with:
    # rows -> name of perturbation
    # columns -> name of each variable that is perturbed
    # values -> the perturbation parameter for each variable
    perturbations = {
        vortex: perturbations[vortex]
        for vortex, number in sorted(
            {
                vortex: int(vortex.split('_')[-1])
                for vortex, pertubations in perturbations.items()
            }.items(),
            key=lambda item: item[1],
        )
    }

    perturbations = DataFrame.from_records(
        list(perturbations.values()), index=list(perturbations)
    )

    return perturbations
