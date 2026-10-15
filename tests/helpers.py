"""Mallas rectangulares estructuradas para los tests."""

from source.mesh.generators import quadrilateral_mesh
from source.mesh.generators import uniform


def rectangle_mesh(width, height, nx, ny, element_type, material, thickness=1.0):
    """Rectángulo [0, width] x [0, height] con nx x ny celdas.

    Cada celda es un cuadrilátero o dos triángulos. Define:
      NodeSets: bottom, top, left, right, boundary (y las esquinas)
      SideSets: bottom, top, left, right
    """
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]

    return quadrilateral_mesh(corners, uniform(nx), uniform(ny), element_type,
                              {1: material}, 1, thickness)
