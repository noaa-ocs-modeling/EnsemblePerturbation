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
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from math import exp, inf, sqrt
from os import PathLike
from pathlib import Path
from random import gauss, random

from adcircpy.forcing.winds.best_track import BestTrackForcing
from dateutil.parser import parse as parse_date
from numpy import append, floor, insert, interp, sign, transpose
from pandas import DataFrame
from pyproj import Proj
from shapely.geometry import LineString

AIR_DENSITY = 1.15  # [kg/m3]

UNIT_CONVERSIONS = {
    'knot': {'meter_per_second': 0.514444444},
    'millibar': {'pascal': 100},
    'pascal': {'millibar': 0.01},
    'nautical_mile': {'statute_mile': 1.150781, 'meter': 1852},
    'statute_mile': {'nautical_mile': 0.868976},
}

E1 = exp(1.0)  # e

# Index of absolute errors (forecast times [hrs)]
ERROR_INDICES_NO_60H = [0, 12, 24, 36, 48, 72, 96, 120]  # no 60-hr data
ERROR_INDICES_60HR = [0, 12, 24, 36, 48, 60, 72, 96, 120]  # has 60-hr data (for Rmax)


class RandomType(Enum):
    GAUSSIAN = 'gaussian'
    LINEAR = 'linear'


class BestTrackPerturbedVariable(ABC):
    name: str
    random_type: RandomType = None
    lower_bound: float = None
    upper_bound: float = None
    historical_forecast_errors: {str: float} = None
    default: float = None  # [mbar]


class CentralPressure(BestTrackPerturbedVariable):
    name = 'central_pressure'


class BackgroundPressure(BestTrackPerturbedVariable):
    name = 'background_pressure'
    default = 1013.0


class MaximumSustainedWindSpeed(BestTrackPerturbedVariable):
    name = 'max_sustained_wind_speed'
    random_type = RandomType.GAUSSIAN
    lower_bound = 25  # [kt]
    upper_bound = 165  # [kt]
    historical_forecast_errors = {
        '<50kt': DataFrame(
            data=[1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [kt]'],
        ),
        '50-95kt': DataFrame(
            data=[2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [kt]'],
        ),
        '>95kt': DataFrame(
            data=[2.80, 7.94, 11.53, 13.27, 12.66, 13.41, 13.46, 13.55],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [kt]'],
        ),
    }


class RadiusOfMaximumWinds(BestTrackPerturbedVariable):
    name = 'radius_of_maximum_winds'
    random_type = RandomType.LINEAR
    lower_bound = 5  # [nm]
    upper_bound = 200  # [nm]
    historical_forecast_errors = {
        '<15sm': DataFrame(
            data=transpose(
                [
                    [0.0, -13.82, -19.67, -21.37, -26.31, -32.71, -39.12, -46.80, -52.68],
                    [0.0, 1.27, 0.22, 1.02, 0.00, -2.59, -5.18, -7.15, -12.91],
                ]
            )
            * UNIT_CONVERSIONS['statute_mile']['nautical_mile'],
            index=ERROR_INDICES_60HR,
            columns=['minimum error [nm]', 'maximum error [nm]'],
        ),
        '15-25sm': DataFrame(
            data=transpose(
                [
                    [0.0, -10.47, -14.54, -20.35, -23.88, -21.78, -19.68, -24.24, -28.30],
                    [0.0, 4.17, 6.70, 6.13, 6.54, 6.93, 7.32, 9.33, 8.03],
                ]
            )
            * UNIT_CONVERSIONS['statute_mile']['nautical_mile'],
            index=ERROR_INDICES_60HR,
            columns=['minimum error [nm]', 'maximum error [nm]'],
        ),
        '25-35sm': DataFrame(
            data=transpose(
                [
                    [0.0, -8.57, -13.41, -10.87, -9.26, -9.34, -9.42, -7.41, -7.40],
                    [0.0, 8.21, 10.62, 13.93, 15.62, 16.04, 16.46, 16.51, 16.70],
                ]
            )
            * UNIT_CONVERSIONS['statute_mile']['nautical_mile'],
            index=ERROR_INDICES_60HR,
            columns=['minimum error [nm]', 'maximum error [nm]'],
        ),
        '35-45sm': DataFrame(
            data=transpose(
                [
                    [0.0, -10.66, -7.64, -5.68, -3.25, -1.72, -0.19, 3.65, 2.59],
                    [0.0, 14.77, 17.85, 22.07, 27.60, 27.08, 26.56, 26.80, 28.30],
                ]
            )
            * UNIT_CONVERSIONS['statute_mile']['nautical_mile'],
            index=ERROR_INDICES_60HR,
            columns=['minimum error [nm]', 'maximum error [nm]'],
        ),
        '>45sm': DataFrame(
            data=transpose(
                [
                    [0.0, -15.36, -10.37, 3.14, 12.10, 12.21, 12.33, 6.66, 7.19],
                    [0.0, 21.43, 29.96, 37.22, 39.27, 39.10, 38.93, 34.40, 35.93],
                ]
            )
            * UNIT_CONVERSIONS['statute_mile']['nautical_mile'],
            index=ERROR_INDICES_60HR,
            columns=['minimum error [nm]', 'maximum error [nm]'],
        ),
    }


class CrossTrack(BestTrackPerturbedVariable):
    name = 'cross_track'
    random_type = RandomType.GAUSSIAN
    lower_bound = -inf
    upper_bound = +inf
    historical_forecast_errors = {
        '<50kt': DataFrame(
            data=[1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [nm]'],
        ),
        '50-95kt': DataFrame(
            data=[2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [nm]'],
        ),
        '>95kt': DataFrame(
            data=[2.80, 7.94, 11.53, 13.27, 12.66, 13.41, 13.46, 13.55],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [nm]'],
        ),
    }


class AlongTrack(BestTrackPerturbedVariable):
    name = 'along_track'
    random_type = RandomType.GAUSSIAN
    lower_bound = -inf
    upper_bound = +inf
    historical_forecast_errors = {
        '<50kt': DataFrame(
            data=[1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [nm]'],
        ),
        '50-95kt': DataFrame(
            data=[2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [nm]'],
        ),
        '>95kt': DataFrame(
            data=[2.80, 7.94, 11.53, 13.27, 12.66, 13.41, 13.46, 13.55],
            index=ERROR_INDICES_NO_60H,
            columns=['mean error [nm]'],
        ),
    }


class BestTrackPerturber:
    def __init__(
        self,
        storm: str,
        nws: int = None,
        interval: timedelta = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ):
        """
        build storm perturber

        :param storm: NHC storm code, for instance `al062018`
        :param nws: wind stress parameter
        :param interval: time interval
        :param start_date: start time of ensemble
        :param end_date: end time of ensemble
        """

        self.storm = storm
        self.nws = nws
        self.interval = interval
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
    def nws(self) -> int:
        return self.__nws

    @nws.setter
    def nws(self, nws: int):
        self.__nws = nws

    @property
    def interval(self) -> timedelta:
        return self.__interval

    @interval.setter
    def interval(self, interval: timedelta):
        if interval is not None and not isinstance(interval, timedelta):
            self.__interval = timedelta(seconds=interval)
        self.__interval = interval

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
    def forcing(self) -> BestTrackForcing:
        interval = self.interval
        if interval is not None:
            interval = interval / timedelta(seconds=1)

        configuration = {
            'storm': self.storm,
            'nws': self.nws,
            'interval_seconds': interval,
            'start_date': self.start_date,
            'end_date': self.end_date,
        }

        if configuration != self.__previous_configuration:
            self.__forcing = BestTrackForcing(**configuration)
            self.__previous_configuration = configuration

        self.__storm = self.__forcing.storm_id

        return self.__forcing

    def write(
        self,
        number_of_perturbations: int,
        variables: [BestTrackPerturbedVariable],
        directory: PathLike = None,
    ):
        """
        :param number_of_perturbations: number of perturbations to create
        :param variables: list of variable names, any combination of `["max_sustained_wind_speed", "radius_of_maximum_winds", "along_track", "cross_track"]`
        :param directory: directory to which to write
        """

        if number_of_perturbations is not None:
            number_of_perturbations = int(number_of_perturbations)
        if directory is None:
            directory = Path.cwd()
        elif not isinstance(directory, Path):
            directory = Path(directory)

        # write out original fort.22
        self.forcing.write(directory / 'original.22', overwrite=True)

        # Get the initial intensity and size
        storm_strength = self.vmax_intensity_class(
            self.compute_initial(MaximumSustainedWindSpeed.name),
        )
        storm_size = self.rmax_size_class(self.compute_initial(RadiusOfMaximumWinds.name))

        print(f'Initial storm strength: {storm_strength}')
        print(f'Initial storm size: {storm_size}')

        # extracting original dataframe
        df_original = self.forcing.df

        # modifying the central pressure while subsequently changing
        # Vmax using the same Holland B parameter,
        # writing each to a new fort.22
        for variable in variables:
            print(f'writing perturbations for "{variable.name}"')
            # print(min(df_original[var]))
            # print(max(df_original[var]))

            # Make the random pertubations based on the historical forecast errors
            # Interpolate from the given VT to the storm_VT
            # print(forecast_errors[var][Initial_Vmax])
            if issubclass(variable, RadiusOfMaximumWinds):
                storm_classification = storm_size
            else:
                storm_classification = storm_strength

            xp = variable.historical_forecast_errors[storm_classification].index
            yp = variable.historical_forecast_errors[storm_classification].values
            base_errors = [
                interp(self.validation_time / timedelta(hours=1), xp, yp[:, ncol])
                for ncol in range(len(yp[0]))
            ]

            # print(base_errors)

            for perturbation_index in range(1, number_of_perturbations + 1):
                # make a deepcopy to preserve the original dataframe
                df_modified = deepcopy(df_original)

                # get the random perturbation sample
                if variable.random_type == RandomType.GAUSSIAN:
                    alpha = gauss(0, 1) / 0.7979
                    # mean_abs_error = 0.7979 * sigma

                    print(f'Random gaussian variable = {alpha}')

                    # add the error to the variable with bounds to some physical constraints
                    df_modified = self.perturb_bound(
                        df_modified, perturbation=base_errors[0] * alpha, variable=variable,
                    )
                elif variable.random_type == RandomType.LINEAR:
                    alpha = random()

                    print(f'Random number in [0,1) = {alpha}')

                    # subtract the error from the variable with physical constraint bounds
                    df_modified = self.perturb_bound(
                        df_modified,
                        perturbation=-(
                            base_errors[0] * (1.0 - alpha) + base_errors[1] * alpha
                        ),
                        variable=variable,
                    )

                if issubclass(variable, MaximumSustainedWindSpeed):
                    # In case of Vmax need to change the central pressure
                    # incongruence with it (obeying Holland B relationship)
                    df_modified[CentralPressure.name] = self.compute_pc_from_Vmax(df_modified)

                # reset the dataframe
                self.forcing._df = df_modified

                # write out the modified fort.22
                self.forcing.write(
                    directory / f'{variable.name}_{perturbation_index}.22', overwrite=True,
                )

    @property
    def validation_time(self) -> timedelta:
        """ get the validation time of storm """
        return self.forcing.datetime - self.forcing.start_date

    def compute_initial(self, var: str) -> float:
        """ the initial value of the input variable var (Vmax or Rmax) """
        return self.forcing.df[var].iloc[0]

    @property
    def holland_B(self) -> float:
        """ Compute Holland B at each time snap """
        df_test = self.forcing.df
        Vmax = (
            df_test[MaximumSustainedWindSpeed.name]
            * UNIT_CONVERSIONS['knot']['meter_per_second']
        )
        DelP = (
            df_test[BackgroundPressure.name] - df_test[CentralPressure.name]
        ) * UNIT_CONVERSIONS['millibar']['pascal']
        B = Vmax * Vmax * AIR_DENSITY * E1 / DelP
        return B

    def compute_pc_from_Vmax(self, dataframe: DataFrame) -> float:
        """ Compute central pressure from Vmax based on Holland B """
        Vmax = (
            dataframe[MaximumSustainedWindSpeed.name]
            * UNIT_CONVERSIONS['knot']['meter_per_second']
        )
        DelP = Vmax * Vmax * AIR_DENSITY * E1 / self.holland_B
        pc = dataframe[BackgroundPressure.name] - DelP * UNIT_CONVERSIONS['pascal']['millibar']
        return pc

    def perturb_bound(
        self,
        dataframe: DataFrame,
        perturbation: [float],
        variable: BestTrackPerturbedVariable,
    ):
        """ perturbing the variable with physical bounds """
        if issubclass(variable, AlongTrack):
            dataframe = self.interpolate_along_track(
                dataframe, along_track_errors=perturbation
            )
        elif issubclass(variable, CrossTrack):
            dataframe = self.offset_track(dataframe, cross_track_errors=perturbation)
        else:
            test_list = dataframe[variable.name] + perturbation
            bounded_result = [
                min(variable.upper_bound, max(ele, variable.lower_bound)) for ele in test_list
            ]
            dataframe[variable.name] = bounded_result
        return dataframe

    def interpolate_along_track(self, dataframe, along_track_errors: [float]) -> DataFrame:
        """
        interpolate_along_track(df_,VT,along_track_errros)
        Offsets points by a given error/distance by interpolating along the track

        :param dataframe: ATCF dataframe containing track info
        :param along_track_errors: along-track errors for each forecast time (VT)
        :return: updated ATCF dataframe with different longitude latitude locations based on interpolated errors along track
        """

        interp_pts = 5  # maximum number of pts along line for each interpolation

        # Get the coordinates of the track
        track_coords = dataframe[['longitude', 'latitude']].values.tolist()

        VT = (self.validation_time / timedelta(hours=1)).values

        # Extrapolating the track for negative errors at beginning and positive errors at end of track
        for vt_index in range(0, len(VT)):
            if VT[vt_index] == 0 and VT[vt_index + 1] > 0:
                # append point to the beginning for going in negative direction
                p1 = track_coords[vt_index]
                p2 = track_coords[vt_index + 1]
                ps = [
                    p1[0] - interp_pts * (p2[0] - p1[0]),
                    p1[1] - interp_pts * (p2[1] - p1[1]),
                ]
            if VT[vt_index] == VT[-1] and VT[vt_index - 1] < VT[-1]:
                # append point to the end going in positive direction
                p1 = track_coords[vt_index - 1]
                p2 = track_coords[vt_index]
                pe = [
                    p1[0] + interp_pts * (p2[0] - p1[0]),
                    p1[1] + interp_pts * (p2[1] - p1[1]),
                ]

        track_coords.insert(0, ps)
        track_coords.append(pe)

        # adding pseudo-VT times to the ends
        VT = insert(VT, 0, VT[0] - 6)
        VT = append(VT, VT[-1] + 6)

        # print(track_coords)
        # print(VT)
        # print(along_track_errors)

        # loop over all coordinates
        lon_new = list()
        lat_new = list()
        for track_coord_index in range(1, len(track_coords) - 1):
            # get the utm projection for middle longitude
            myProj = utm_proj_from_lon(track_coords[track_coord_index][0])
            along_error = (
                along_track_errors[track_coord_index - 1]
                * UNIT_CONVERSIONS['nautical_mile']['meter']
            )
            along_sign = int(sign(along_error))

            pts = list()
            ind = track_coord_index
            while len(pts) < interp_pts:
                if ind < 0 or ind > len(track_coords) - 1:
                    break  # reached end of line
                if ind == track_coord_index or VT[ind] != VT[ind - along_sign]:
                    # get the x,y utm coordinate for this line string
                    x_utm, y_utm = myProj(
                        track_coords[ind][0], track_coords[ind][1], inverse=False
                    )
                    pts.append((x_utm, y_utm))
                ind = ind + along_sign

            # make the temporary line segment
            line_segment = LineString([pts[pp] for pp in range(0, len(pts))])

            # interpolate a distance "along_error" along the line
            pnew = line_segment.interpolate(abs(along_error))

            # get back lat-lon
            lon, lat = myProj(pnew.coords[0][0], pnew.coords[0][1], inverse=True,)

            # print(track_coords[idx-1:idx+2])
            # print(along_error/111e3)
            # print(new_coords])

            lon_new.append(lon)
            lat_new.append(lat)

        # print([lon_new, lat_new])

        dataframe['longitude'] = lon_new
        dataframe['latitude'] = lat_new

        return dataframe

    def offset_track(self, dataframe, cross_track_errors: [float]) -> DataFrame:
        """
        offset_track(df_,VT,cross_track_errors)
          - Offsets points by a given perpendicular error/distance from the original track

        :param dataframe: ATCF dataframe containing track info
        :param cross_track_errors: cross-track errors [nm] for each forecast time (VT)
        :return: updated ATCF dataframe with different longitude latitude locations based on perpendicular offsets set by the cross_track_errors
        """

        # Get the coordinates of the track
        track_coords = dataframe[['longitude', 'latitude']].values.tolist()

        VT = (self.validation_time / timedelta(hours=1)).values

        # loop over all coordinates
        lon_new = list()
        lat_new = list()
        for track_coord_index in range(0, len(track_coords)):
            # get the current cross_track_error
            cross_error = (
                cross_track_errors[track_coord_index]
                * UNIT_CONVERSIONS['nautical_mile']['meter']
            )

            # get the utm projection for the reference coordinate
            myProj = utm_proj_from_lon(track_coords[track_coord_index][0])

            # get the location of the original reference coordinate
            x_ref, y_ref = myProj(
                track_coords[track_coord_index][0],
                track_coords[track_coord_index][1],
                inverse=False,
            )

            # get the index of the previous forecasted coordinate
            idx_p = track_coord_index - 1
            while idx_p >= 0:
                if VT[idx_p] < VT[track_coord_index]:
                    break
                idx_p = idx_p - 1
            if idx_p < 0:  # beginning of track
                idx_p = track_coord_index

            # get previous projected coordinate
            x_p, y_p = myProj(track_coords[idx_p][0], track_coords[idx_p][1], inverse=False,)

            # get the perpendicular offset based on the line connecting from the previous coordinate to the current coordinate
            dx_p, dy_p = get_offset(x_p, y_p, x_ref, y_ref, cross_error)

            # get the index of the next forecasted coordinate
            idx_n = track_coord_index + 1
            while idx_n < len(track_coords):
                if VT[idx_n] > VT[track_coord_index]:
                    break
                idx_n = idx_n + 1
            if idx_n == len(track_coords):  # end of track
                idx_n = track_coord_index

            # get previous projected coordinate
            x_n, y_n = myProj(track_coords[idx_n][0], track_coords[idx_n][1], inverse=False,)

            # get the perpendicular offset based on the line connecting from the current coordinate to the next coordinate
            dx_n, dy_n = get_offset(x_ref, y_ref, x_n, y_n, cross_error)

            # get the perpendicular offset based on the average of the forward and backward piecewise track lines adjusted so that the distance matches the actual cross_error
            dx = 0.5 * (dx_p + dx_n)
            dy = 0.5 * (dy_p + dy_n)
            alpha = abs(cross_error) / sqrt(dx ** 2 + dy ** 2)

            # compute the next point and retrieve back the lat-lon geographic coordinate
            lon, lat = myProj(x_ref + alpha * dx, y_ref + alpha * dy, inverse=True,)
            lon_new.append(lon)
            lat_new.append(lat)

        dataframe['longitude'] = lon_new
        dataframe['latitude'] = lat_new

        return dataframe

    @staticmethod
    def vmax_intensity_class(vmax: float) -> str:
        """ Category for Vmax based intensity """
        if vmax < 50:
            return '<50kt'  # weak
        elif vmax > 95:
            return '>95kt'  # strong
        else:
            return '50-95kt'  # medium

    @staticmethod
    def rmax_size_class(rmax: float) -> str:
        """ Category for Rmax based size """
        # convert from nautical miles to statute miles
        rmw_sm = rmax * UNIT_CONVERSIONS['nautical_mile']['statute_mile']
        if rmw_sm < 15:
            return '<15sm'  # very small
        elif rmw_sm < 25:
            return '15-25sm'  # small
        elif rmw_sm < 35:
            return '25-35sm'  # medium
        elif rmw_sm < 45:
            return '35-45sm'  # large
        else:
            return '>45sm'  # very large


def utm_proj_from_lon(lon_mean: float) -> Proj:
    """
    utm_from_lon - UTM zone for a longitude
    Not right for some polar regions (Norway, Svalbard, Antartica)
    :param lon_mean: longitude

    :usage   x_utm,y_utm   = myProj(lon, lat  , inverse=False)
    :usage   lon, lat      = myProj(xutm, yutm, inverse=True)
    """

    zone = floor((lon_mean + 180) / 6) + 1
    # print("Zone is " + str(zone))

    return Proj(f'+proj=utm +zone={zone}K, +ellps=WGS84 +datum=WGS84 +units=m +no_defs')


def get_offset(x1: float, y1: float, x2: float, y2: float, d: float,) -> (float, float):
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


if __name__ == '__main__':
    ##################################
    # Example calls from command line for 2018 Hurricane Florence:
    # - python3 make_storm_ensemble.py 3 al062018 2018-09-11-06 2018-09-17-06
    # - python3 make_storm_ensemble.py 5 Florence2018 2018-09-11-06
    ##################################

    # Implement argument parsing
    argument_parser = ArgumentParser()
    argument_parser.add_argument('number_of_perturbations', help='number of perturbations')
    argument_parser.add_argument('storm_code', help='storm name/code')
    argument_parser.add_argument('start_date', nargs='?', help='start date')
    argument_parser.add_argument('end_date', nargs='?', help='end date')
    arguments = argument_parser.parse_args()

    # hardcoding variable list for now
    variables = [
        MaximumSustainedWindSpeed,
        RadiusOfMaximumWinds,
        AlongTrack,
        CrossTrack,
    ]

    perturber = BestTrackPerturber(
        storm=arguments.storm_code,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )

    perturber.write(
        number_of_perturbations=arguments.number_of_perturbations, variables=variables,
    )
