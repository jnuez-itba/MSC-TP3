"""Geometría y datos de la presa de gravedad.

Sistema de coordenadas: x hacia aguas abajo, y hacia arriba. El paramento de
aguas arriba llega a la cresta en x = 0. El agua está del lado izquierdo.

             crest
           |<--->|
        (0,H)    (crest, H)
          |        \\
          |         \\
          |          \\
        (-a,0)--------(base - a, 0)
        talón          pie
"""


class DamGeometry:

    def __init__(self, height=30.0, base=18.0, crest=3.0, freeboard=2.0,
                 upstream_batter=0.0, concrete_density=2400.0,
                 water_density=1000.0):
        self.height = height
        self.base = base
        self.crest = crest
        self.freeboard = freeboard
        self.upstream_batter = upstream_batter
        self.concrete_density = concrete_density
        self.water_density = water_density

    def water_level(self):
        return self.height - self.freeboard

    def heel(self):
        """Talón (esquina aguas arriba de la base)."""
        return (-self.upstream_batter, 0.0)

    def toe(self):
        """Pie (esquina aguas abajo de la base)."""
        return (self.base - self.upstream_batter, 0.0)

    def corners(self):
        """Esquinas en sentido antihorario: talón, pie, cresta aguas abajo,
        cresta aguas arriba."""
        return [self.heel(), self.toe(), (self.crest, self.height), (0.0, self.height)]

    def area(self):
        return polygon_area(self.corners())

    def centroid(self):
        return polygon_centroid(self.corners())

    def description(self):
        return ("H = {:g} m, base = {:g} m, cresta = {:g} m, borde libre = {:g} m, "
                "talud aguas arriba = {:g} m").format(
                    self.height, self.base, self.crest, self.freeboard,
                    self.upstream_batter)


def polygon_area(points):
    A = 0.0
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        A = A + x1 * y2 - x2 * y1

    return A / 2.0


def polygon_centroid(points):
    A = polygon_area(points)
    cx = 0.0
    cy = 0.0
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx = cx + (x1 + x2) * cross
        cy = cy + (y1 + y2) * cross

    return cx / (6.0 * A), cy / (6.0 * A)
