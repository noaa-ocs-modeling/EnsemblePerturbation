from shapely.geometry import Point


def parse_stations(filename: str) -> {str: Point}:
    with open(filename) as stations_file:
        lines = [line.split()
                 for line in list(stations_file.readlines())[2:]]
        return {int(line[3].replace('! ', '')):
                    Point(float(line[0]), float(line[1]))
                for line in lines}
