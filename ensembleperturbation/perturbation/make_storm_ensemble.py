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
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timedelta
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


def main(
        number_of_perturbations: int,
        variable_list: [str],
        storm_code: str,
        start_date: datetime,
        end_date: datetime,
        output_directory: PathLike = None,
):
    """
    Write perturbed tracks to `fort.22`

    :param number_of_perturbations: number of perturbations to create
    :param variable_list: list of variable names, any combination of `["max_sustained_wind_speed", "radius_of_maximum_winds", "along_track", "cross_track"]`
    :param storm_code: NHC storm code, for instance `al062018`
    :param start_date: start time of ensemble
    :param end_date: end time of ensemble
    :param output_directory: directory to which to write
    """

    if output_directory is None:
        output_directory = Path.cwd()
    elif not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    # getting best track
    best_track = BestTrackForcing(
        storm_code,
        start_date=start_date,
        end_date=end_date,
    )

    # write out original fort.22
    best_track.write(
        output_directory / 'original.22',
        overwrite=True,
    )

    # Computing Holland B and validation times from BT
    holland_B = compute_Holland_B(best_track)
    storm_VT = compute_VT_hours(best_track)

    # Get the initial intensity and size
    storm_strength = intensity_class(
        compute_initial(best_track, vmax_variable),
    )
    storm_size = size_class(compute_initial(best_track, rmw_var))

    print(f'Initial storm strength: {storm_strength}')
    print(f'Initial storm size: {storm_size}')

    # extracting original dataframe
    df_original = best_track.df

    # modifying the central pressure while subsequently changing
    # Vmax using the same Holland B parameter,
    # writing each to a new fort.22
    for variable in variable_list:
        print(f'writing perturbations for "{variable}"')
        # print(min(df_original[var]))
        # print(max(df_original[var]))

        # Make the random pertubations based on the historical forecast errors
        # Interpolate from the given VT to the storm_VT
        # print(forecast_errors[var][Initial_Vmax])
        if variable == 'radius_of_maximum_winds':
            storm_classification = storm_size
        else:
            storm_classification = storm_strength

        xp = forecast_errors[variable][storm_classification].index
        yp = forecast_errors[variable][storm_classification].values
        base_errors = [
            interp(storm_VT, xp, yp[:, ncol])
            for ncol in range(len(yp[0]))
        ]

        # print(base_errors)

        for perturbation_index in range(1, number_of_perturbations + 1):
            # make a deepcopy to preserve the original dataframe
            df_modified = deepcopy(df_original)

            # get the random perturbation sample
            if random_variable_type[variable] == 'gauss':
                alpha = gauss(0, 1) / 0.7979
                # mean_abs_error = 0.7979 * sigma

                print(f'Random gaussian variable = {alpha}')

                # add the error to the variable with bounds to some physical constraints
                df_modified = perturb_bound(
                    df_modified,
                    perturbation=base_errors[0] * alpha,
                    variable=variable,
                    validation_time=storm_VT,
                )
            elif random_variable_type[variable] == 'range':
                alpha = random()

                print(f'Random number in [0,1) = {alpha}')

                # subtract the error from the variable with physical constraint bounds
                df_modified = perturb_bound(
                    df_modified,
                    perturbation=-(base_errors[0] * (1.0 - alpha) +
                                   base_errors[1] * alpha),
                    variable=variable,
                )

            if variable == vmax_variable:
                # In case of Vmax need to change the central pressure
                # incongruence with it (obeying Holland B relationship)
                df_modified[pc_var] = compute_pc_from_Vmax(
                    df_modified,
                    B=holland_B,
                )

            # reset the dataframe
            best_track._df = df_modified

            # write out the modified fort.22
            best_track.write(
                output_directory / f'{variable}_{perturbation_index}.22',
                overwrite=True,
            )


################################################################
## Sub functions and dictionaries...
################################################################
# get the validation time of storm in hours
def compute_VT_hours(best_track: BestTrackForcing) -> float:
    return (best_track.datetime - best_track.start_date) / \
           timedelta(hours=1)


# the initial value of the input variable var (Vmax or Rmax)
def compute_initial(best_track: BestTrackForcing, var: str) -> float:
    return best_track.df[var].iloc[0]


# some constants
rho_air = 1.15  # density of air [kg/m3]
Pb = 1013.0  # background pressure [mbar]
kts2ms = 0.514444444  # kts to m/s
mbar2pa = 100  # mbar to Pa
pa2mbar = 0.01  # Pa to mbar
nm2sm = 1.150781  # nautical miles to statute miles
sm2nm = 0.868976  # statute miles to nautical miles
e1 = exp(1.0)  # e
# variable names
pc_var = 'central_pressure'
pb_var = 'background_pressure'
vmax_variable = 'max_sustained_wind_speed'
rmw_var = 'radius_of_maximum_winds'


# Compute Holland B at each time snap
def compute_Holland_B(best_track: BestTrackForcing) -> float:
    df_test = best_track.df
    Vmax = df_test[vmax_variable] * kts2ms
    DelP = (df_test[pb_var] - df_test[pc_var]) * mbar2pa
    B = Vmax * Vmax * rho_air * e1 / DelP
    return B


# Compute central pressure from Vmax based on Holland B
def compute_pc_from_Vmax(dataframe: DataFrame, B: float) -> float:
    Vmax = dataframe[vmax_variable] * kts2ms
    DelP = Vmax * Vmax * rho_air * e1 / B
    pc = dataframe[pb_var] - DelP * pa2mbar
    return pc


# random variable types (Gaussian or just a range)
random_variable_type = {
    'max_sustained_wind_speed': 'gauss',
    'radius_of_maximum_winds': 'range',
    'cross_track': 'gauss',
    'along_track': 'gauss',
}
# physical bounds of different variables
lower_bound = {
    'max_sustained_wind_speed': 25,  # [kt]
    'radius_of_maximum_winds': 5,  # [nm]
    'cross_track': -inf,
    'along_track': -inf,
}
upper_bound = {
    'max_sustained_wind_speed': 165,  # [kt]
    'radius_of_maximum_winds': 200,  # [nm]
    'cross_track': +inf,
    'along_track': +inf,
}


# perturbing the variable with physical bounds
def perturb_bound(
        dataframe: DataFrame,
        perturbation: float,
        variable: str,
        validation_time: float = None,
):
    if variable == 'along_track':
        dataframe = interpolate_along_track(
            dataframe,
            VT=validation_time.values,
            along_track_errors=perturbation,
        )
    elif variable == 'cross_track':
        dataframe = offset_track(
            dataframe,
            VT=validation_time.values,
            cross_track_errors=perturbation,
        )
    else:
        test_list = dataframe[variable] + perturbation
        LB = lower_bound[variable]
        UB = upper_bound[variable]
        bounded_result = [min(UB, max(ele, LB)) for ele in test_list]
        dataframe[variable] = bounded_result
    return dataframe


# Category for Vmax based intensity
def intensity_class(vmax: float) -> str:
    if vmax < 50:
        return '<50kt'  # weak
    elif vmax > 95:
        return '>95kt'  # strong
    else:
        return '50-95kt'  # medium


# Category for Rmax based size
def size_class(rmw_nm: float) -> str:
    # convert from nautical miles to statute miles
    rmw_sm = rmw_nm * nm2sm
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


# Index of absolute errors (forecast times [hrs)]
VT = [0, 12, 24, 36, 48, 72, 96, 120]  # no 60-hr data
VTR = [0, 12, 24, 36, 48, 60, 72, 96, 120]  # has 60-hr data (for Rmax)
# Mean absolute Vmax errors based on initial intensity
Vmax_weak_errors = DataFrame(
    data=[1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91],
    index=VT,
    columns=['mean error [kt]'],
)
Vmax_medium_errors = DataFrame(
    data=[2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62],
    index=VT,
    columns=['mean error [kt]'],
)
Vmax_strong_errors = DataFrame(
    data=[2.80, 7.94, 11.53, 13.27, 12.66, 13.41, 13.46, 13.55],
    index=VT,
    columns=['mean error [kt]'],
)
# RMW errors bound based on initial size
RMW_vsmall_errors = DataFrame(
    data=sm2nm * transpose(
        [
            [0.0, -13.82, -19.67, -21.37, -26.31, -32.71, -39.12,
             -46.80, -52.68],
            [0.0, 1.27, 0.22, 1.02, 0.00, -2.59, -5.18, -7.15, -12.91],
        ]
    ),
    index=VTR,
    columns=['minimum error [nm]', 'maximum error [nm]'],
)
RMW_small_errors = DataFrame(
    data=sm2nm * transpose(
        [
            [0.0, -10.47, -14.54, -20.35, -23.88, -21.78, -19.68,
             -24.24, -28.30],
            [0.0, 4.17, 6.70, 6.13, 6.54, 6.93, 7.32, 9.33, 8.03],
        ]
    ),
    index=VTR,
    columns=['minimum error [nm]', 'maximum error [nm]'],
)
RMW_medium_errors = DataFrame(
    data=sm2nm * transpose(
        [
            [0.0, -8.57, -13.41, -10.87, -9.26, -9.34, -9.42, -7.41,
             -7.40],
            [0.0, 8.21, 10.62, 13.93, 15.62, 16.04, 16.46, 16.51,
             16.70],
        ]
    ),
    index=VTR,
    columns=['minimum error [nm]', 'maximum error [nm]'],
)
RMW_large_errors = DataFrame(
    data=sm2nm * transpose(
        [
            [0.0, -10.66, -7.64, -5.68, -3.25, -1.72, -0.19, 3.65,
             2.59],
            [0.0, 14.77, 17.85, 22.07, 27.60, 27.08, 26.56, 26.80,
             28.30],
        ]
    ),
    index=VTR,
    columns=['minimum error [nm]', 'maximum error [nm]'],
)
RMW_vlarge_errors = DataFrame(
    data=sm2nm * transpose(
        [
            [0.0, -15.36, -10.37, 3.14, 12.10, 12.21, 12.33, 6.66,
             7.19],
            [0.0, 21.43, 29.96, 37.22, 39.27, 39.10, 38.93, 34.40,
             35.93],
        ]
    ),
    index=VTR,
    columns=['minimum error [nm]', 'maximum error [nm]'],
)
# Mean absolute cross-track errors based on initial intensity
ct_weak_errors = DataFrame(
    data=[1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91],
    index=VT,
    columns=['mean error [nm]'],
)
ct_medium_errors = DataFrame(
    data=[2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62],
    index=VT,
    columns=['mean error [nm]'],
)
ct_strong_errors = DataFrame(
    data=[2.80, 7.94, 11.53, 13.27, 12.66, 13.41, 13.46, 13.55],
    index=VT,
    columns=['mean error [nm]'],
)
# Mean absolute along-track errors based on initial intensity
at_weak_errors = DataFrame(
    data=[1.45, 4.01, 6.17, 8.42, 10.46, 14.28, 18.26, 19.91],
    index=VT,
    columns=['mean error [nm]'],
)
at_medium_errors = DataFrame(
    data=[2.26, 5.75, 8.54, 9.97, 11.28, 13.11, 13.46, 12.62],
    index=VT,
    columns=['mean error [nm]'],
)
at_strong_errors = DataFrame(
    data=[2.80, 7.94, 11.53, 13.27, 12.66, 13.41, 13.46, 13.55],
    index=VT,
    columns=['mean error [nm]'],
)
# Dictionary of historical forecast errors by variable
forecast_errors = {
    'max_sustained_wind_speed': {
        '<50kt': Vmax_weak_errors,
        '50-95kt': Vmax_medium_errors,
        '>95kt': Vmax_strong_errors,
    },
    'radius_of_maximum_winds': {
        '<15sm': RMW_vsmall_errors,
        '15-25sm': RMW_small_errors,
        '25-35sm': RMW_medium_errors,
        '35-45sm': RMW_large_errors,
        '>45sm': RMW_vlarge_errors,
    },
    'cross_track': {
        '<50kt': ct_weak_errors,
        '50-95kt': ct_medium_errors,
        '>95kt': ct_strong_errors,
    },
    'along_track': {
        '<50kt': at_weak_errors,
        '50-95kt': at_medium_errors,
        '>95kt': at_strong_errors,
    },
}


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

    return Proj(
        f'+proj=utm +zone={zone}K, +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    )


def interpolate_along_track(
        df_,
        VT: [float],
        along_track_errors: [float],
) -> DataFrame:
    """
    interpolate_along_track(df_,VT,along_track_errros)
    Offsets points by a given error/distance by interpolating along the track

    :param df_: ATCF dataframe containing track info
    :param VT: the forecast validation times [hours]
    :param along_track_errors: along-track errors for each forecast time (VT)
    :return: updated ATCF dataframe with different longitude latitude locations based on interpolated errors along track
    """

    # Parameters
    interp_pts = 5  # maximum number of pts along line for each interpolation
    nm2m = 1852  # nautical miles to meters

    # Get the coordinates of the track
    track_coords = df_[['longitude', 'latitude']].values.tolist()

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
        along_error = along_track_errors[track_coord_index - 1] * nm2m
        along_sign = int(sign(along_error))

        pts = list()
        ind = track_coord_index
        while len(pts) < interp_pts:
            if ind < 0 or ind > len(track_coords) - 1:
                break  # reached end of line
            if ind == track_coord_index or VT[ind] != VT[
                ind - along_sign]:
                # get the x,y utm coordinate for this line string
                x_utm, y_utm = myProj(
                    track_coords[ind][0], track_coords[ind][1],
                    inverse=False
                )
                pts.append((x_utm, y_utm))
            ind = ind + along_sign

        # make the temporary line segment
        line_segment = LineString([
            pts[pp] for pp in range(0, len(pts))
        ])

        # interpolate a distance "along_error" along the line
        pnew = line_segment.interpolate(abs(along_error))

        # get back lat-lon
        lon, lat = myProj(
            pnew.coords[0][0],
            pnew.coords[0][1],
            inverse=True,
        )

        # print(track_coords[idx-1:idx+2])
        # print(along_error/111e3)
        # print(new_coords])

        lon_new.append(lon)
        lat_new.append(lat)

    # print([lon_new, lat_new])

    df_['longitude'] = lon_new
    df_['latitude'] = lat_new

    return df_


def get_offset(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        d: float,
) -> (float, float):
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


def offset_track(
        df_,
        VT: [float],
        cross_track_errors: [float],
) -> DataFrame:
    """
    offset_track(df_,VT,cross_track_errors)
      - Offsets points by a given perpendicular error/distance from the original track

    :param df_: ATCF dataframe containing track info
    :param VT: the forecast validation times [hours]
    :param cross_track_errors: cross-track errors [nm] for each forecast time (VT)
    :return: updated ATCF dataframe with different longitude latitude locations based on perpendicular offsets set by the cross_track_errors
    """

    # Parameters
    nm2m = 1852  # nautical miles to meters

    # Get the coordinates of the track
    track_coords = df_[['longitude', 'latitude']].values.tolist()

    # loop over all coordinates
    lon_new = list()
    lat_new = list()
    for track_coord_index in range(0, len(track_coords)):
        # get the current cross_track_error
        cross_error = cross_track_errors[track_coord_index] * nm2m

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
        x_p, y_p = myProj(
            track_coords[idx_p][0],
            track_coords[idx_p][1],
            inverse=False,
        )

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
        x_n, y_n = myProj(
            track_coords[idx_n][0],
            track_coords[idx_n][1],
            inverse=False,
        )

        # get the perpendicular offset based on the line connecting from the current coordinate to the next coordinate
        dx_n, dy_n = get_offset(x_ref, y_ref, x_n, y_n, cross_error)

        # get the perpendicular offset based on the average of the forward and backward piecewise track lines adjusted so that the distance matches the actual cross_error
        dx = 0.5 * (dx_p + dx_n)
        dy = 0.5 * (dy_p + dy_n)
        alpha = abs(cross_error) / sqrt(dx ** 2 + dy ** 2)

        # compute the next point and retrieve back the lat-lon geographic coordinate
        lon, lat = myProj(
            x_ref + alpha * dx,
            y_ref + alpha * dy,
            inverse=True,
        )
        lon_new.append(lon)
        lat_new.append(lat)

    df_['longitude'] = lon_new
    df_['latitude'] = lat_new

    return df_


if __name__ == '__main__':
    ##################################
    # Example calls from command line for 2018 Hurricane Florence:
    # - python3 make_storm_ensemble.py 3 al062018 2018-09-11-06 2018-09-17-06
    # - python3 make_storm_ensemble.py 5 Florence2018 2018-09-11-06
    ##################################

    # Implement argument parsing
    argument_parser = ArgumentParser()
    argument_parser.add_argument('number_of_perturbations',
                                 help='number of perturbations')
    argument_parser.add_argument('storm_code', help='storm name/code')
    argument_parser.add_argument('start_date', nargs='?',
                                 help='start date')
    argument_parser.add_argument('end_date', nargs='?', help='end date')
    arguments = argument_parser.parse_args()

    # Parse number of perturbations
    num = arguments.number_of_perturbations
    if num is not None:
        num = int(num)

    # Parse storm code
    stormcode = arguments.storm_code

    # Parse the start and end dates, e.g., YYYY-MM-DD-HH
    start_date = arguments.start_date
    if start_date is not None:
        start_date = parse_date(start_date)
    end_date = arguments.end_date
    if end_date is not None:
        end_date = parse_date(end_date)

    # hardcoding variable list for now
    variables = [
        'max_sustained_wind_speed',
        'radius_of_maximum_winds',
        'along_track',
        'cross_track',
    ]

    # Enter function
    main(num, variables, stormcode, start_date, end_date)
