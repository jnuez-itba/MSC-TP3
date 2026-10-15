"""Generadores de mallas estructuradas.

quadrilateral_mesh: malla de una región cuadrilátera mediante un mapeo
bilineal del cuadrado unitario (xi, eta) en [0, 1] x [0, 1]:

    P(xi, eta) = (1-xi)(1-eta) P1 + xi (1-eta) P2 + xi eta P3 + (1-xi) eta P4

con P1..P4 las esquinas en sentido antihorario. Las líneas de xi constante
y de eta constante son rectas, así que los nodos de lado de los elementos
cuadráticos quedan en el punto medio de cada lado.

dam_mesh: sección de la presa de gravedad (usa quadrilateral_mesh).
"""

import math

from source.mesh.mesh import Mesh
from source.mesh.mesh import ELEMENT_NAMES
from source.mesh.node import Node


FAMILY_ORDER = {
    "TRI3": ("TRI", 1),
    "TRI6": ("TRI", 2),
    "QUAD4": ("QUAD", 1),
    "QUAD9": ("QUAD", 2),
}


def refine(values, order):
    """Para elementos cuadráticos agrega el punto medio entre cada par de
    valores consecutivos (los nodos de lado y centrales)."""
    if order == 1:
        return list(values)

    refined = []

    for k in range(len(values) - 1):
        refined.append(values[k])
        refined.append((values[k] + values[k + 1]) / 2.0)

    refined.append(values[-1])

    return refined


def uniform(n):
    return [k / n for k in range(n + 1)]


def quadrilateral_mesh(corners, xi_values, eta_values, element_type,
                       materials, material_id=1, thickness=1.0):
    """Malla estructurada de un cuadrilátero.

    corners     4 esquinas (x, y) en sentido antihorario: P1 P2 P3 P4.
                El lado P1-P2 es el "bottom", P2-P3 "right", P3-P4 "top" y
                P4-P1 "left".
    xi_values   posiciones de las líneas de la grilla en [0, 1] a lo largo
                de P1-P2 (deben empezar en 0 y terminar en 1)
    eta_values  ídem a lo largo de P1-P4
    element_type  TRI3, QUAD4, TRI6 o QUAD9 (los triángulos se arman
                  partiendo cada celda por la diagonal)
    materials   diccionario {id: material}

    Define:
      NodeSets: bottom, right, top, left, boundary, y las 4 esquinas
                corner_1 ... corner_4
      SideSets: bottom, right, top, left
    """
    if element_type not in FAMILY_ORDER:
        raise ValueError("Unknown element type: " + element_type)

    family, order = FAMILY_ORDER[element_type]

    xi_grid = refine(xi_values, order)
    eta_grid = refine(eta_values, order)

    nx = len(xi_values) - 1
    ny = len(eta_values) - 1
    columns = len(xi_grid)
    rows = len(eta_grid)

    p1, p2, p3, p4 = corners

    mesh = Mesh()
    mesh.dim = 2
    mesh.dofs_per_node = 2
    mesh.thickness = thickness
    mesh.materials = dict(materials)

    def node_id(i, j):
        return j * columns + i

    for j in range(rows):
        eta = eta_grid[j]
        for i in range(columns):
            xi = xi_grid[i]

            x = ((1 - xi) * (1 - eta) * p1[0] + xi * (1 - eta) * p2[0]
                 + xi * eta * p3[0] + (1 - xi) * eta * p4[0])
            y = ((1 - xi) * (1 - eta) * p1[1] + xi * (1 - eta) * p2[1]
                 + xi * eta * p3[1] + (1 - xi) * eta * p4[1])

            mesh.nodes.append(Node(node_id(i, j), x, y))

    sides = {"bottom": [], "right": [], "top": [], "left": []}

    def add_element(connectivity):
        mesh.element_global_ids.append(len(mesh.connectivity))
        mesh.element_types.append(ELEMENT_NAMES[(family, order)])
        mesh.element_families.append(family)
        mesh.element_orders.append(order)
        mesh.element_material_ids.append(material_id)
        mesh.connectivity.append(connectivity)
        return len(mesh.connectivity) - 1

    p = order

    for ey in range(ny):
        for ex in range(nx):
            i0 = p * ex
            j0 = p * ey

            a = node_id(i0, j0)
            b = node_id(i0 + p, j0)
            c = node_id(i0 + p, j0 + p)
            d = node_id(i0, j0 + p)

            if family == "QUAD":
                if order == 1:
                    e = add_element([a, b, c, d])
                else:
                    e = add_element([
                        a, b, c, d,
                        node_id(i0 + 1, j0), node_id(i0 + 2, j0 + 1),
                        node_id(i0 + 1, j0 + 2), node_id(i0, j0 + 1),
                        node_id(i0 + 1, j0 + 1),
                    ])

                border = [(e, 0, ey == 0, "bottom"), (e, 1, ex == nx - 1, "right"),
                          (e, 2, ey == ny - 1, "top"), (e, 3, ex == 0, "left")]

            else:
                # celda partida en dos triángulos por la diagonal a-c
                if order == 1:
                    e1 = add_element([a, b, c])
                    e2 = add_element([a, c, d])
                else:
                    e1 = add_element([
                        a, b, c,
                        node_id(i0 + 1, j0), node_id(i0 + 2, j0 + 1),
                        node_id(i0 + 1, j0 + 1),
                    ])
                    e2 = add_element([
                        a, c, d,
                        node_id(i0 + 1, j0 + 1), node_id(i0 + 1, j0 + 2),
                        node_id(i0, j0 + 1),
                    ])

                border = [(e1, 0, ey == 0, "bottom"), (e1, 1, ex == nx - 1, "right"),
                          (e2, 1, ey == ny - 1, "top"), (e2, 2, ex == 0, "left")]

            for element_id, side, on_border, name in border:
                if on_border:
                    sides[name].append((element_id, side))

    mesh.side_sets = sides

    mesh.node_sets = {
        "bottom": [node_id(i, 0) for i in range(columns)],
        "right": [node_id(columns - 1, j) for j in range(rows)],
        "top": [node_id(i, rows - 1) for i in range(columns)],
        "left": [node_id(0, j) for j in range(rows)],
        "corner_1": [node_id(0, 0)],
        "corner_2": [node_id(columns - 1, 0)],
        "corner_3": [node_id(columns - 1, rows - 1)],
        "corner_4": [node_id(0, rows - 1)],
    }

    boundary = set()
    for name in ["bottom", "right", "top", "left"]:
        boundary.update(mesh.node_sets[name])

    mesh.node_sets["boundary"] = sorted(boundary)

    return mesh


def dam_mesh(material, element_type="QUAD9", element_size=1.0,
             height=30.0, base=18.0, crest=3.0, freeboard=2.0,
             upstream_batter=0.0, thickness=1.0):
    """Malla de la sección de la presa de gravedad.

    Geometría (x hacia aguas abajo, y hacia arriba, origen en el pie del
    paramento de aguas arriba cuando es vertical):

             crest
           |<--->|
        (0,H)    (crest, H)
          |        \\
          |         \\         agua del lado izquierdo (aguas arriba),
          |          \\        nivel = H - freeboard
          |           \\
        (-a,0)--------(base-a, 0)
          |<--- base --->|

    upstream_batter  a: corrimiento horizontal del talón aguas arriba.
                     0 = paramento vertical (perfil del enunciado).

    La grilla tiene una fila de nodos exactamente en el nivel del agua, para
    que la presión hidrostática (que tiene un quiebre ahí) se integre exacto.

    NodeSets:  base, heel (talón aguas arriba), toe (pie aguas abajo),
               upstream, downstream, crest
    SideSets:  base, upstream, upstream_wet (bajo el agua), downstream, crest
    """
    H = height
    a = upstream_batter
    water_level = H - freeboard

    if not (0.0 < freeboard < H):
        raise ValueError("Expected 0 < freeboard < height")

    if crest <= 0.0 or base <= crest:
        raise ValueError("Expected 0 < crest < base")

    corners = [(-a, 0.0), (base - a, 0.0), (crest, H), (0.0, H)]

    # Columnas: según el ancho medio de la sección
    mean_width = (base + crest) / 2.0
    nx = max(1, int(math.ceil(mean_width / element_size)))

    # Filas: un tramo bajo el agua y otro en el borde libre, para que haya
    # una fila de nodos en el nivel del agua
    ny_water = max(1, int(round(water_level / element_size)))
    ny_dry = max(1, int(round(freeboard / element_size)))

    eta_values = [k * water_level / ny_water / H for k in range(ny_water)]
    eta_values += [(water_level + k * freeboard / ny_dry) / H for k in range(ny_dry + 1)]

    mesh = quadrilateral_mesh(corners, uniform(nx), eta_values, element_type,
                              {1: material}, 1, thickness)

    # Nombres propios de la presa
    rename_nodes = {"bottom": "base", "right": "downstream", "top": "crest",
                    "left": "upstream", "corner_1": "heel", "corner_2": "toe"}
    rename_sides = {"bottom": "base", "right": "downstream", "top": "crest",
                    "left": "upstream"}

    mesh.node_sets = {rename_nodes[k]: v for k, v in mesh.node_sets.items()
                      if k in rename_nodes}
    mesh.side_sets = {rename_sides[k]: v for k, v in mesh.side_sets.items()}

    # Lados del paramento aguas arriba que están bajo el agua
    wet = []
    tolerance = 1.0e-9 * H

    for element_id, side in mesh.side_sets["upstream"]:
        connectivity = mesh.connectivity[element_id]
        family = mesh.element_families[element_id]
        n_vertices = 3 if family == "TRI" else 4

        start = mesh.node(connectivity[side])
        end = mesh.node(connectivity[(side + 1) % n_vertices])

        if max(start.y, end.y) <= water_level + tolerance:
            wet.append((element_id, side))

    mesh.side_sets["upstream_wet"] = wet

    return mesh
