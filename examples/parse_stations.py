from os import PathLike
from pathlib import Path

import fiona
from pyproj import CRS
from shapely.geometry import Point, mapping

from ensemble_perturbation.utilities import repository_root


def parse_stations(filename: PathLike) -> {str: Point}:
    with open(filename) as stations_file:
        lines = [line.split() for line in list(stations_file.readlines())[2:]]
        return {
            int(line[3].replace('! ', '')): Point(float(line[0]), float(line[1]))
            for line in lines
        }


if __name__ == '__main__':
    stations_filename = Path(repository_root()) / 'examples/data/stations.txt'
    stations = parse_stations(stations_filename)

    stations_vector_filename = Path(repository_root()) / 'examples/data/stations.gpkg'
    schema = {'geometry': 'Point', 'properties': {'id': 'str'}}
    crs = CRS.from_epsg(4326)
    with fiona.open(
            stations_vector_filename, 'w', 'GPKG', schema, crs.to_dict()
    ) as stations_vector_file:
        stations_vector_file.writerecords(
            [
                {'geometry': mapping(point), 'properties': {'id': id}}
                for id, point in stations.items()
            ]
        )

    print('done')
