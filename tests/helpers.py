"""Mallas rectangulares estructuradas para los tests."""

from source.mesh.mesh import Mesh
from source.mesh.mesh import ELEMENT_NAMES
from source.mesh.node import Node


FAMILY_ORDER = {
    "TRI3": ("TRI", 1),
    "TRI6": ("TRI", 2),
    "QUAD4": ("QUAD", 1),
    "QUAD9": ("QUAD", 2),
}


def rectangle_mesh(width, height, nx, ny, element_type, material, thickness=1.0):
    """Rectángulo [0, width] x [0, height] con nx x ny celdas.

    Cada celda es un cuadrilátero o dos triángulos. Define:
      NodeSets: bottom, top, left, right, boundary
      SideSets: bottom, top, left, right
    """
    family, order = FAMILY_ORDER[element_type]
    p = order

    columns = p * nx + 1
    rows = p * ny + 1

    mesh = Mesh()
    mesh.dim = 2
    mesh.dofs_per_node = 2
    mesh.thickness = thickness
    mesh.materials = {1: material}

    def node_id(i, j):
        return j * columns + i

    for j in range(rows):
        for i in range(columns):
            x = width * i / (columns - 1)
            y = height * j / (rows - 1)
            mesh.nodes.append(Node(node_id(i, j), x, y))

    sides = {"bottom": [], "top": [], "left": [], "right": []}

    def add_element(connectivity):
        mesh.element_global_ids.append(len(mesh.connectivity))
        mesh.element_types.append(ELEMENT_NAMES[(family, order)])
        mesh.element_families.append(family)
        mesh.element_orders.append(order)
        mesh.element_material_ids.append(1)
        mesh.connectivity.append(connectivity)
        return len(mesh.connectivity) - 1

    for ey in range(ny):
        for ex in range(nx):
            i0 = p * ex
            j0 = p * ey

            a = node_id(i0, j0)
            b = node_id(i0 + p, j0)
            c = node_id(i0 + p, j0 + p)
            d = node_id(i0, j0 + p)

            if element_type == "QUAD4":
                e = add_element([a, b, c, d])
                quad_sides = [(e, 0, ey == 0, "bottom"), (e, 1, ex == nx - 1, "right"),
                              (e, 2, ey == ny - 1, "top"), (e, 3, ex == 0, "left")]

            elif element_type == "QUAD9":
                e = add_element([
                    a, b, c, d,
                    node_id(i0 + 1, j0), node_id(i0 + 2, j0 + 1),
                    node_id(i0 + 1, j0 + 2), node_id(i0, j0 + 1),
                    node_id(i0 + 1, j0 + 1),
                ])
                quad_sides = [(e, 0, ey == 0, "bottom"), (e, 1, ex == nx - 1, "right"),
                              (e, 2, ey == ny - 1, "top"), (e, 3, ex == 0, "left")]

            elif element_type == "TRI3":
                e1 = add_element([a, b, c])
                e2 = add_element([a, c, d])
                quad_sides = [(e1, 0, ey == 0, "bottom"), (e1, 1, ex == nx - 1, "right"),
                              (e2, 1, ey == ny - 1, "top"), (e2, 2, ex == 0, "left")]

            elif element_type == "TRI6":
                e1 = add_element([
                    a, b, c,
                    node_id(i0 + 1, j0), node_id(i0 + 2, j0 + 1), node_id(i0 + 1, j0 + 1),
                ])
                e2 = add_element([
                    a, c, d,
                    node_id(i0 + 1, j0 + 1), node_id(i0 + 1, j0 + 2), node_id(i0, j0 + 1),
                ])
                quad_sides = [(e1, 0, ey == 0, "bottom"), (e1, 1, ex == nx - 1, "right"),
                              (e2, 1, ey == ny - 1, "top"), (e2, 2, ex == 0, "left")]

            for element_id, side, on_border, name in quad_sides:
                if on_border:
                    sides[name].append((element_id, side))

    mesh.side_sets = sides

    mesh.node_sets = {
        "bottom": [node_id(i, 0) for i in range(columns)],
        "top": [node_id(i, rows - 1) for i in range(columns)],
        "left": [node_id(0, j) for j in range(rows)],
        "right": [node_id(columns - 1, j) for j in range(rows)],
    }

    boundary = set()
    for name in mesh.node_sets:
        boundary.update(mesh.node_sets[name])

    mesh.node_sets["boundary"] = sorted(boundary)

    return mesh
