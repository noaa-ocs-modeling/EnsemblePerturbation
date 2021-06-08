from netCDF4._netCDF4 import Variable
import numpy


def decode_time(variable: Variable, unit: str = None) -> numpy.array:
    if unit is None:
        unit = variable.units
    unit, direction, base_date = unit.split(' ', 2)
    intervals = {
        'years': 'Y',
        'months': 'M',
        'days': 'D',
        'hours': 'h',
        'minutes': 'm',
        'seconds': 's',
    }
    base_date = base_date.strip(' UTC')
    return numpy.datetime64(base_date) + numpy.array(variable).astype(
        f'timedelta64[{intervals[unit]}]'
    )
