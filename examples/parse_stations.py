from pathlib import Path

import fiona
from pyproj import CRS
from shapely.geometry import mapping

from ensemble_perturbation import repository_root
from ensemble_perturbation.outputs.parse_stations import parse_stations

if __name__ == '__main__':
    stations_filename = Path(repository_root()) / 'examples/data/stations.txt'
    stations = parse_stations(stations_filename)

    stations_vector_filename = Path(repository_root()) / \
                               'examples/data/stations.gpkg'
    schema = {'geometry': 'Point', 'properties': {'id': 'str'}}
    crs = CRS.from_epsg(4326)
    with fiona.open(stations_vector_filename, 'w', 'GPKG', schema,
                    crs.to_dict()) as stations_vector_file:
        stations_vector_file.writerecords([
            {'geometry': mapping(point), 'properties': {'id': id}}
            for id, point in stations.items()
        ])

    print('done')
