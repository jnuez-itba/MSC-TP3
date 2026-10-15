"""Modelo de elementos finitos de la presa.

solve_dam arma y resuelve el problema con:
  - peso propio del hormigón (densidad del material de la malla);
  - presión hidrostática sobre el SideSet "upstream_wet";
  - condición de borde en la base según 'support':

      "fixed"   empotramiento: u = v = 0 en toda la base.
                Fundación rígida (roca mucho más rígida que el hormigón) con
                adherencia perfecta: equivale a fricción infinita. No hay
                asentamiento, deslizamiento ni despegue.

      "roller"  apoyo deslizante sin fricción (rodillo): v = 0 en toda la
                base, u = 0 solo en el talón (para impedir el movimiento de
                cuerpo rígido). Fundación rígida con fricción nula. Todo el
                empuje horizontal lo toma un único nodo.

      "robin"   condición de Robin: resortes distribuidos en la base con
                rigidez normal k_normal y tangencial k_tangential [N/m3].
                Fundación elástica (Winkler). Es un caso intermedio entre
                los dos anteriores.

  La fricción de Coulomb (adherencia hasta |t| = mu sigma, después
  deslizamiento) es no lineal y queda fuera de este modelo lineal: "fixed" y
  "roller" son los dos casos límite.
"""

import numpy

from source.elements.factory import create_elements
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.systems.boundary_conditions import fix_node_set
from source.systems.loads import GRAVITY
from source.systems.loads import add_self_weight
from source.systems.loads import add_side_pressure
from source.systems.loads import hydrostatic_pressure
from source.systems.reactions import compute_reactions
from source.systems.reactions import node_set_reaction_resultant
from source.systems.robin import add_robin_side
from source.solvers.linear import solve


SUPPORTS = ["fixed", "roller", "robin"]


def solve_dam(mesh, water_level, support="fixed", water_density=1000.0,
              gravity=GRAVITY, k_normal=None, k_tangential=None):
    """Resuelve la presa. Devuelve un diccionario con la malla, los
    elementos, los desplazamientos u, las reacciones R y su resultante
    respecto del pie (Fx, Fy, Mz; Mz positivo antihorario).

    Con support="robin" hay que dar k_normal y k_tangential.
    """
    if support not in SUPPORTS:
        raise ValueError("Unknown support: " + support)

    if support == "robin" and (k_normal is None or k_tangential is None):
        raise ValueError("Robin support needs k_normal and k_tangential")

    mesh.dofs_per_node = 2
    elements = create_elements(mesh)

    K = DenseMatrix(mesh.number_of_dofs())
    for element in elements:
        K.scatter_local_to_global(element)

    f = numpy.zeros(mesh.number_of_dofs())
    add_self_weight(f, elements, gravity)
    add_side_pressure(f, mesh, elements, "upstream_wet",
                      hydrostatic_pressure(water_level, water_density, gravity))

    # K y f de la estructura sola, antes de los apoyos: con ellos las
    # reacciones son R = K u - f (en Robin, la fuerza de los resortes)
    K_original = K.values.copy()
    f_original = f.copy()

    if support == "fixed":
        fix_node_set(K, f, mesh, "base", (0, 1))

    elif support == "roller":
        fix_node_set(K, f, mesh, "base", (1,))
        heel = mesh.node_sets["heel"][0]
        apply_dirichlet_bc(K, f, mesh.node_global_dofs(heel)[0], 0.0)

    else:
        add_robin_side(K, mesh, elements, "base", k_normal, k_tangential)

    u = solve(K, f)

    R = compute_reactions(K_original, f_original, u)

    toe = mesh.node(mesh.node_sets["toe"][0])
    Fx, Fy, Mz = node_set_reaction_resultant(mesh, R, "base", (toe.x, toe.y))

    return {
        "mesh": mesh,
        "elements": elements,
        "u": u,
        "reactions": R,
        "reaction_x": Fx,
        "reaction_y": Fy,
        "reaction_moment_toe": Mz,
        "support": support,
    }


def base_nodes_sorted(mesh):
    """Ids de los nodos de la base ordenados por x."""
    ids = list(mesh.node_sets["base"])
    ids.sort(key=lambda i: mesh.node(i).x)
    return ids
