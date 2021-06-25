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
from datetime import datetime, timedelta
from enum import Enum
from math import exp, inf, sqrt
from os import PathLike
from pathlib import Path
from random import gauss, random
from typing import Union

from dateutil.parser import parse as parse_date
import numpy
from numpy import floor, interp, sign
from pandas import DataFrame, Series
import pint
from pint_pandas import PintType
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection
from shapely.geometry import LineString

from ensembleperturbation.tropicalcyclone.atcf import VortexForcing
from ensembleperturbation.utilities import units

AIR_DENSITY = 1.15 * units.kilogram / units.meters ** 3

E1 = exp(1.0)  # e

# Index of absolute errors (forecast times [hrs)]
ERROR_INDICES_NO_60H = [0, 12, 24, 36, 48, 72, 96, 120]  # no 60-hr data
ERROR_INDICES_60HR = [0, 12, 24, 36, 48, 60, 72, 96, 120]  # has 60-hr data (for Rmax)


class PerturbationType(Enum):
    GAUSSIAN = 'gaussian'
    LINEAR = 'linear'


class VortexPerturbedVariable(ABC):
    name: str
    perturbation_type: PerturbationType

    def __init__(
        self,
        lower_bound: float = None,
        upper_bound: float = None,
        historical_forecast_errors: {str: DataFrame} = None,
        default: float = None,
        unit: pint.Unit = None,
    ):
        self.__unit = None
        self.__lower_bound = None
        self.__upper_bound = None
        self.__historical_forecast_errors = None
        self.__default = None

        self.unit = unit

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.historical_forecast_errors = historical_forecast_errors
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
    def lower_bound(self) -> pint.Quantity:
        if self.__lower_bound.units != self.unit:
            self.__lower_bound.ito(self.unit)
        return self.__lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound: float):
        if isinstance(lower_bound, pint.Quantity):
            if lower_bound.units != self.unit:
                lower_bound = lower_bound.to(self.unit)
        elif lower_bound is not None:
            lower_bound *= self.unit
        self.__lower_bound = lower_bound

    @property
    def upper_bound(self) -> pint.Quantity:
        if self.__upper_bound.units != self.unit:
            self.__upper_bound.ito(self.unit)
        return self.__upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound: float):
        if isinstance(upper_bound, pint.Quantity):
            if upper_bound.units != self.unit:
                upper_bound = upper_bound.to(self.unit)
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

    @property
    def default(self) -> pint.Quantity:
        if self.__default is not None and self.__default.units != self.unit:
            self.__default.ito(self.unit)
        return self.__default

    @default.setter
    def default(self, default: float):
        if isinstance(default, pint.Quantity):
            if default.units != self.unit:
                default = default.to(self.unit)
        elif default is not None:
            default *= self.unit
        self.__default = default

    def perturb(
        self, vortex_dataframe: DataFrame, values: [float], times: [datetime],
    ) -> DataFrame:
        """
        perturb the variable within physical bounds

        :param vortex_dataframe: ATCF dataframe containing track info
        :param values: values for each forecast time (VT)
        :param times: forecast times (VT)
        :return: updated ATCF dataframe with perturbed values
        """

        all_values = vortex_dataframe[self.name].values + values
        vortex_dataframe[self.name] = [
            min(self.upper_bound, max(value, self.lower_bound)).magnitude
            for value in all_values
        ] * self.unit

        return vortex_dataframe

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(lower_bound={repr(self.lower_bound)}, upper_bound={repr(self.upper_bound)}, historical_forecast_errors={repr(self.historical_forecast_errors)}, default={repr(self.default)}, unit={repr(self.unit)})'


class CentralPressure(VortexPerturbedVariable):
    name = 'central_pressure'


class BackgroundPressure(VortexPerturbedVariable):
    name = 'background_pressure'

    def __init__(self):
        super().__init__(
            default=1013.0, unit=units.millibar,
        )


class MaximumSustainedWindSpeed(VortexPerturbedVariable):
    name = 'max_sustained_wind_speed'
    perturbation_type = PerturbationType.GAUSSIAN

    def __init__(self):
        super().__init__(
            lower_bound=25,
            upper_bound=165,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {'mean error [kt]': [1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91]},
                    index=ERROR_INDICES_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {'mean error [kt]': [2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62]},
                    index=ERROR_INDICES_NO_60H,
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
                    index=ERROR_INDICES_NO_60H,
                ),
            },
            unit=units.knot,
        )


class RadiusOfMaximumWinds(VortexPerturbedVariable):
    name = 'radius_of_maximum_winds'
    perturbation_type = PerturbationType.LINEAR

    def __init__(self):
        super().__init__(
            lower_bound=5,
            upper_bound=200,
            historical_forecast_errors={
                '<15sm': DataFrame(
                    {
                        'minimum error [nm]': Series(
                            [
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
                            dtype=PintType(units.us_statute_mile),
                        ),
                        'maximum error [nm]': Series(
                            [0.0, 1.27, 0.22, 1.02, 0.00, -2.59, -5.18, -7.15, -12.91],
                            dtype=PintType(units.us_statute_mile),
                        ),
                    },
                    index=ERROR_INDICES_60HR,
                ),
                '15-25sm': DataFrame(
                    {
                        'minimum error [nm]': Series(
                            [
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
                            dtype=PintType(units.us_statute_mile),
                        ),
                        'maximum error [nm]': Series(
                            [0.0, 4.17, 6.70, 6.13, 6.54, 6.93, 7.32, 9.33, 8.03],
                            dtype=PintType(units.us_statute_mile),
                        ),
                    },
                    index=ERROR_INDICES_60HR,
                ),
                '25-35sm': DataFrame(
                    {
                        'minimum error [nm]': Series(
                            [0.0, -8.57, -13.41, -10.87, -9.26, -9.34, -9.42, -7.41, -7.40],
                            dtype=PintType(units.us_statute_mile),
                        ),
                        'maximum error [nm]': Series(
                            [0.0, 8.21, 10.62, 13.93, 15.62, 16.04, 16.46, 16.51, 16.70],
                            dtype=PintType(units.us_statute_mile),
                        ),
                    },
                    index=ERROR_INDICES_60HR,
                ),
                '35-45sm': DataFrame(
                    {
                        'minimum error [nm]': Series(
                            [0.0, -10.66, -7.64, -5.68, -3.25, -1.72, -0.19, 3.65, 2.59],
                            index=ERROR_INDICES_60HR,
                            dtype=PintType(units.us_statute_mile),
                        ),
                        'maximum error [nm]': Series(
                            [0.0, 14.77, 17.85, 22.07, 27.60, 27.08, 26.56, 26.80, 28.30],
                            index=ERROR_INDICES_60HR,
                            dtype=PintType(units.us_statute_mile),
                        ),
                    },
                ),
                '>45sm': DataFrame(
                    {
                        'minimum error [nm]': Series(
                            [0.0, -15.36, -10.37, 3.14, 12.10, 12.21, 12.33, 6.66, 7.19],
                            dtype=PintType(units.us_statute_mile),
                        ),
                        'maximum error [nm]': Series(
                            [0.0, 21.43, 29.96, 37.22, 39.27, 39.10, 38.93, 34.40, 35.93],
                            dtype=PintType(units.us_statute_mile),
                        ),
                    },
                    index=ERROR_INDICES_60HR,
                ),
            },
            unit=units.nautical_mile,
        )


class CrossTrack(VortexPerturbedVariable):
    name = 'cross_track'
    perturbation_type = PerturbationType.GAUSSIAN

    def __init__(self):
        super().__init__(
            lower_bound=-inf,
            upper_bound=+inf,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {'mean error [nm]': [1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91]},
                    index=ERROR_INDICES_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {'mean error [nm]': [2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62]},
                    index=ERROR_INDICES_NO_60H,
                ),
                '>95kt': DataFrame(
                    {
                        'mean error [nm]': [
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
                    index=ERROR_INDICES_NO_60H,
                ),
            },
            unit=units.nautical_mile,
        )

    def perturb(
        self, vortex_dataframe: DataFrame, values: [float], times: [datetime],
    ) -> DataFrame:
        """
        offset_track(df_,VT,cross_track_errors)
          - Offsets points by a given perpendicular error/distance from the original track

        :param vortex_dataframe: ATCF dataframe containing track info
        :param values: cross-track errors [nm] for each forecast time (VT)
        :param times: forecast times (VT)
        :return: updated ATCF dataframe with different longitude latitude locations based on perpendicular offsets set by the cross_track_errors
        """

        # Get the coordinates of the track
        track_coords = vortex_dataframe[['longitude', 'latitude']].values.tolist()

        # get the utm projection for the reference coordinate
        utm_crs = utm_crs_from_longitude(numpy.mean(track_coords[0]))
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(wgs84, utm_crs)

        times = (times / timedelta(hours=1)).values * units.hours

        # loop over all coordinates
        new_coordinates = []
        for track_coord_index in range(0, len(track_coords)):
            # get the current cross_track_error
            cross_track_error = values[track_coord_index].to(units.meter)

            # get the location of the original reference coordinate
            x_ref, y_ref = (
                transformer.transform(
                    track_coords[track_coord_index][0], track_coords[track_coord_index][1],
                )
                * units.meter
            )

            # get the index of the previous forecasted coordinate
            idx_p = track_coord_index - 1
            while idx_p >= 0:
                if times[idx_p] < times[track_coord_index]:
                    break
                idx_p = idx_p - 1
            if idx_p < 0:  # beginning of track
                idx_p = track_coord_index

            # get previous projected coordinate
            x_p, y_p = (
                transformer.transform(track_coords[idx_p][0], track_coords[idx_p][1])
                * units.meter
            )

            # get the perpendicular offset based on the line connecting from the previous coordinate to the current coordinate
            dx_p, dy_p = get_offset(x_p, y_p, x_ref, y_ref, cross_track_error)

            # get the index of the next forecasted coordinate
            idx_n = track_coord_index + 1
            while idx_n < len(track_coords):
                if times[idx_n] > times[track_coord_index]:
                    break
                idx_n = idx_n + 1
            if idx_n == len(track_coords):  # end of track
                idx_n = track_coord_index

            # get previous projected coordinate
            x_n, y_n = (
                transformer.transform(track_coords[idx_n][0], track_coords[idx_n][1])
                * units.meter
            )

            # get the perpendicular offset based on the line connecting from the current coordinate to the next coordinate
            dx_n, dy_n = get_offset(x_ref, y_ref, x_n, y_n, cross_track_error)

            # get the perpendicular offset based on the average of the forward and backward piecewise track lines adjusted so that the distance matches the actual cross_error
            dx = 0.5 * (dx_p + dx_n)
            dy = 0.5 * (dy_p + dy_n)
            alpha = (abs(cross_track_error) / numpy.sqrt(dx ** 2 + dy ** 2)).magnitude

            # compute the next point and retrieve back the lat-lon geographic coordinate
            new_coordinates.append(
                transformer.transform(
                    (x_ref + alpha * dx).magnitude,
                    (y_ref + alpha * dy).magnitude,
                    direction=TransformDirection.INVERSE,
                )
            )

        degree = PintType(units.degree)
        vortex_dataframe['longitude'], vortex_dataframe['latitude'] = zip(*new_coordinates)
        vortex_dataframe['longitude'] = vortex_dataframe['longitude'].astype(
            degree, copy=False
        )
        vortex_dataframe['latitude'] = vortex_dataframe['latitude'].astype(degree, copy=False)

        return vortex_dataframe


class AlongTrack(VortexPerturbedVariable):
    name = 'along_track'
    perturbation_type = PerturbationType.GAUSSIAN

    def __init__(self):
        super().__init__(
            lower_bound=-inf,
            upper_bound=+inf,
            historical_forecast_errors={
                '<50kt': DataFrame(
                    {'mean error [nm]': [1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91]},
                    index=ERROR_INDICES_NO_60H,
                ),
                '50-95kt': DataFrame(
                    {'mean error [nm]': [2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62]},
                    index=ERROR_INDICES_NO_60H,
                ),
                '>95kt': DataFrame(
                    {
                        'mean error [nm]': [
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
                    index=ERROR_INDICES_NO_60H,
                ),
            },
            unit=units.nautical_mile,
        )

    def perturb(
        self, vortex_dataframe: DataFrame, values: [float], times: [datetime],
    ) -> DataFrame:
        """
        interpolate_along_track(df_,VT,along_track_errros)
        Offsets points by a given error/distance by interpolating along the track

        :param vortex_dataframe: ATCF dataframe containing track info
        :param values: along-track errors for each forecast time (VT)
        :param times: forecast timed (VT)
        :return: updated ATCF dataframe with different longitude latitude locations based on interpolated errors along track
        """

        max_interpolated_points = 5  # maximum number of pts along line for each interpolation

        # Get the coordinates of the track
        coordinates = vortex_dataframe[['longitude', 'latitude']].values

        # get the utm projection for the reference coordinate
        utm_crs = utm_crs_from_longitude(numpy.mean(coordinates[:, 0]))
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(wgs84, utm_crs)

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
            * numpy.expand_dims(numpy.arange(1, max_interpolated_points + 1), axis=1)
        )
        after_diffs = numpy.repeat(
            [unique_points[-1] - unique_points[-2]], max_interpolated_points, axis=0
        ) * numpy.expand_dims(numpy.arange(1, max_interpolated_points + 1), axis=1)
        coordinates = numpy.concatenate(
            [coordinates[0] - previous_diffs, coordinates, coordinates[-1] + after_diffs,]
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
        for index in range(len(values)):
            along_error = values[index - 1].to(units.meter)
            along_sign = int(sign(along_error))

            projected_points = []
            track_index = index
            while len(projected_points) < max_interpolated_points:
                if track_index < 0 or track_index > len(coordinates) - 1:
                    break  # reached end of line
                if (
                    track_index == index
                    or hours[track_index] != hours[track_index - along_sign]
                ):
                    # get the x,y utm coordinate for this line string
                    projected_points.append(
                        transformer.transform(
                            coordinates[track_index][0], coordinates[track_index][1],
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

        degree = PintType(units.degree)
        vortex_dataframe['longitude'], vortex_dataframe['latitude'] = zip(*new_coordinates)
        vortex_dataframe['longitude'] = vortex_dataframe['longitude'].astype(
            degree, copy=False
        )
        vortex_dataframe['latitude'] = vortex_dataframe['latitude'].astype(degree, copy=False)

        return vortex_dataframe


class VortexPerturber:
    def __init__(
        self, storm: str, start_date: datetime = None, end_date: datetime = None,
    ):
        """
        build storm perturber

        :param storm: NHC storm code, for instance `al062018`
        :param start_date: start time of ensemble
        :param end_date: end time of ensemble
        """

        self.storm = storm
        self.start_date = start_date
        self.end_date = end_date

        self.__forcing = None
        self.__previous_configuration = None

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
    def forcing(self) -> VortexForcing:

        configuration = {
            'storm': self.storm,
            'start_date': self.start_date,
            'end_date': self.end_date,
        }

        if configuration != self.__previous_configuration:
            self.__forcing = VortexForcing(**configuration)
            self.__previous_configuration = configuration

        self.__storm = self.__forcing.storm_id

        return self.__forcing

    def write(
        self,
        number_of_perturbations: int,
        variables: [VortexPerturbedVariable],
        directory: PathLike = None,
        alpha: float = None,
    ) -> [Path]:
        """
        :param number_of_perturbations: number of perturbations to create
        :param variables: list of variable names, any combination of `["max_sustained_wind_speed", "radius_of_maximum_winds", "along_track", "cross_track"]`
        :param directory: directory to which to write
        :param alpha: value between [0, 1) with which to multiply error for perturbation; leave None for random
        """

        if number_of_perturbations is not None:
            number_of_perturbations = int(number_of_perturbations)

        for index, variable in enumerate(variables):
            if isinstance(variable, type):
                variables[index] = variable()

        if directory is None:
            directory = Path.cwd()
        elif not isinstance(directory, Path):
            directory = Path(directory)

        # write out original fort.22
        self.forcing.write(directory / 'original.22', overwrite=True)

        # Get the initial intensity and size
        storm_strength = storm_intensity_class(
            self.compute_initial(MaximumSustainedWindSpeed.name),
        )
        storm_size = storm_size_class(self.compute_initial(RadiusOfMaximumWinds.name))

        print(f'Initial storm strength: {storm_strength}')
        print(f'Initial storm size: {storm_size}')

        # extracting original dataframe
        original_data = self.forcing.data

        # add units to data frame
        original_data = original_data.astype(
            {
                variable.name: PintType(variable.unit)
                for variable in variables
                if variable.name in original_data
            },
            copy=False,
        )

        # for each variable, perturb the values and write each to a new `fort.22`
        output_filenames = []
        for variable in variables:
            print(f'writing perturbations for "{variable.name}"')
            # Make the random pertubations based on the historical forecast errors
            # Interpolate from the given VT to the storm_VT
            if isinstance(variable, RadiusOfMaximumWinds):
                storm_classification = storm_size
            else:
                storm_classification = storm_strength

            historical_forecast_errors = variable.historical_forecast_errors[
                storm_classification
            ]
            for column in historical_forecast_errors:
                if isinstance(historical_forecast_errors[column].dtype, PintType):
                    historical_forecast_errors[column] = historical_forecast_errors[
                        column
                    ].pint.magnitude

            xp = historical_forecast_errors.index
            fp = historical_forecast_errors.values
            base_errors = [
                interp(self.validation_times / timedelta(hours=1), xp, fp[:, ncol])
                for ncol in range(len(fp[0]))
            ]

            for perturbation_index in range(1, number_of_perturbations + 1):
                # make a deepcopy to preserve the original dataframe
                perturbed_data = original_data.copy(deep=True)
                perturbed_data = perturbed_data.astype(
                    {
                        variable.name: PintType(variable.unit)
                        for variable in variables
                        if variable.name in original_data
                    },
                    copy=False,
                )

                # get the random perturbation sample
                if variable.perturbation_type == PerturbationType.GAUSSIAN:
                    if alpha is None:
                        alpha = gauss(0, 1) / 0.7979

                    print(f'Random gaussian variable = {alpha}')
                    perturbation = base_errors[0] * alpha
                    if variable.unit is not None and variable.unit != units.dimensionless:
                        perturbation *= variable.unit

                    # add the error to the variable with bounds to some physical constraints
                    perturbed_data = variable.perturb(
                        perturbed_data, values=perturbation, times=self.validation_times,
                    )
                elif variable.perturbation_type == PerturbationType.LINEAR:
                    if alpha is None:
                        alpha = random()

                    print(f'Random number in [0,1) = {alpha}')
                    perturbation = -(base_errors[0] * (1.0 - alpha) + base_errors[1] * alpha)
                    if variable.unit is not None and variable.unit != units.dimensionless:
                        perturbation *= variable.unit

                    # subtract the error from the variable with physical constraint bounds
                    perturbed_data = variable.perturb(
                        perturbed_data, values=perturbation, times=self.validation_times,
                    )

                if isinstance(variable, MaximumSustainedWindSpeed):
                    # In case of Vmax need to change the central pressure incongruence with it (obeying Holland B relationship)
                    perturbed_data[CentralPressure.name] = self.compute_pc_from_Vmax(
                        perturbed_data
                    )

                # remove units from data frame
                for column in perturbed_data:
                    if isinstance(perturbed_data[column].dtype, PintType):
                        perturbed_data[column] = perturbed_data[column].pint.magnitude

                # reset the dataframe
                self.forcing.dataframe = perturbed_data

                # write out the modified fort.22
                output_filename = directory / f'{variable.name}_{perturbation_index}.22'
                self.forcing.write(
                    output_filename, overwrite=True,
                )
                output_filenames.append(output_filename)

        return output_filenames

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


def storm_intensity_class(max_sustained_wind_speed: float) -> str:
    """
    Category for Vmax based intensity

    :param max_sustained_wind_speed: maximum sustained wind speed, in knots
    :return: intensity classification
    """

    if not isinstance(max_sustained_wind_speed, pint.Quantity):
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

    if not isinstance(radius_of_maximum_winds, pint.Quantity):
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


def utm_crs_from_longitude(longitude: float) -> CRS:
    """
    utm_from_lon - UTM zone for a longitude
    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param longitude: longitude
    :return: coordinate reference system
    """

    return CRS.from_epsg(32600 + int(floor((longitude + 180) / 6) + 1))


def get_offset(x1: float, y1: float, x2: float, y2: float, d: float) -> (float, float):
    """
    get_offset(x1,y1,x2,y2,d)
      - get the perpendicular offset to the line (x1,y1) -> (x2,y2) by a distance of d
    """

    if x1 == x2:
        dx = d
        dy = 0
    elif y1 == y2:
        dy = d
        dx = 0
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

        # tangential slope approximation
        slope = (y2 - y1) / (x2 - x1)

        # normal slope
        pslope = -1 / slope  # (might be 1/slope depending on direction of travel)
        psign = ((pslope > 0) == (x1 > x2)) * 2 - 1

        dx = psign * d / sqrt(1 + pslope ** 2)
        dy = pslope * dx

    return dx, dy
