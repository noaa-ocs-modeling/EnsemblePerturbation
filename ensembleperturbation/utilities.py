from concurrent.futures import ProcessPoolExecutor
import logging
from os import PathLike
from pathlib import Path
import sys
import traceback

import numpy
import pint
from pint_pandas import PintType
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point

units = pint.UnitRegistry()
PintType.ureg = units


def repository_root(path: PathLike = None) -> Path:
    if path is None:
        path = __file__
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_file():
        path = path.parent
    if '.git' in (child.name for child in path.iterdir()) or path == path.parent:
        return path
    else:
        return repository_root(path.parent)


def get_logger(
    name: str,
    log_filename: PathLike = None,
    file_level: int = None,
    console_level: int = None,
    log_format: str = None,
) -> logging.Logger:
    """
    instantiate logger instance

    :param name: name of logger
    :param log_filename: path to log file
    :param file_level: minimum log level to write to log file
    :param console_level: minimum log level to print to console
    :param log_format: logger message format
    :return: instance of a Logger object
    """

    if file_level is None:
        file_level = logging.DEBUG
    if console_level is None:
        console_level = logging.INFO
    logger = logging.getLogger(name)

    # check if logger is already configured
    if logger.level == logging.NOTSET and len(logger.handlers) == 0:
        # check if logger has a parent
        if '.' in name:
            if isinstance(logger.parent, logging.RootLogger):
                for existing_console_handler in [
                    handler
                    for handler in logger.parent.handlers
                    if not isinstance(handler, logging.FileHandler)
                ]:
                    logger.parent.removeHandler(existing_console_handler)
            logger.parent = get_logger(name.rsplit('.', 1)[0])
        else:
            # otherwise create a new split-console logger
            if console_level != logging.NOTSET:
                for existing_console_handler in [
                    handler
                    for handler in logger.handlers
                    if not isinstance(handler, logging.FileHandler)
                ]:
                    logger.removeHandler(existing_console_handler)

                console_output = logging.StreamHandler(sys.stdout)
                console_output.setLevel(console_level)
                logger.addHandler(console_output)

    if log_filename is not None:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(file_level)
        for existing_file_handler in [
            handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)
        ]:
            logger.removeHandler(existing_file_handler)
        logger.addHandler(file_handler)

    if log_format is None:
        log_format = '[%(asctime)s] %(name)-15s %(levelname)-8s: %(message)s'
    log_formatter = logging.Formatter(log_format)
    for handler in logger.handlers:
        handler.setFormatter(log_formatter)

    return logger


def ellipsoidal_distance(
    point_a: (float, float), point_b: (float, float), crs_a: CRS, crs_b: CRS = None
) -> float:
    if isinstance(point_a, Point):
        point_a = [*point_a.coords]
    if isinstance(point_b, Point):
        point_b = [*point_b.coords]
    if crs_b is not None:
        transformer = Transformer.from_crs(crs_b, crs_a)
        point_b = transformer.transform(*point_b)
    points = numpy.stack((point_a, point_b), axis=0)
    ellipsoid = crs_a.datum.to_json_dict()['ellipsoid']
    geodetic = Geod(a=ellipsoid['semi_major_axis'], rf=ellipsoid['inverse_flattening'])
    return geodetic.line_length(points[:, 0], points[:, 1])


class ProcessPoolExecutorStackTraced(ProcessPoolExecutor):
    def submit(self, fn, *args, **kwargs):
        """Submits the wrapped function instead of `fn`"""

        return super(ProcessPoolExecutorStackTraced, self).submit(
            self._function_wrapper, fn, *args, **kwargs,
        )

    @staticmethod
    def _function_wrapper(fn, *args, **kwargs):
        """
        Wraps `fn` in order to preserve the traceback of any kind of raised exception
        """

        try:
            return fn(*args, **kwargs)
        except Exception:
            # Creates an exception of the same type with the traceback as message
            raise sys.exc_info()[0](traceback.format_exc())


def encode_categorical_values(values: list, unique_values: list = None) -> list:
    if unique_values is None:
        unique_values = numpy.unique(values)

    if not isinstance(values, numpy.ndarray) or len(values.shape) == 1:
        return numpy.concatenate(
            [numpy.where(unique_values == value) for value in values]
        ).squeeze()
    else:
        rows = []
        for row in values:
            rows.append(encode_categorical_values(row, unique_values=unique_values))
        return numpy.stack(rows, axis=0)
