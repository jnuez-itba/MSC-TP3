"""Condición de borde de Robin: resortes distribuidos sobre un SideSet.

La tracción que hace el apoyo es proporcional al desplazamiento:

    t = -(k_normal (u . n) n + k_tangential (u . s) s)

Representa una fundación elástica (modelo de Winkler): el apoyo no está fijo
ni libre. Cada lado del SideSet suma a K una matriz de resortes
(Element2D.side_spring_matrix). Como no se impone ningún desplazamiento, los
resortes son los que impiden el movimiento de cuerpo rígido.

Límites:
    k -> infinito          base empotrada
    k_tangential -> 0      apoyo deslizante sin fricción (rodillo)
"""


def scatter_matrix(K, element, Ke):
    dofs = element.global_dofs()

    for i in range(len(dofs)):
        for j in range(len(dofs)):
            if Ke[i][j] != 0.0:
                K.add(dofs[i], dofs[j], Ke[i][j])


def add_robin_side(K, mesh, elements, side_set_name, k_normal, k_tangential):
    """Suma a K los resortes de todos los lados de un SideSet."""
    if side_set_name not in mesh.side_sets:
        raise ValueError("Unknown SideSet: " + side_set_name)

    for element_id, side in mesh.side_sets[side_set_name]:
        element = elements[element_id]
        Ks = element.side_spring_matrix(side, k_normal, k_tangential)
        scatter_matrix(K, element, Ks)


def winkler_stiffness(E_foundation, nu_foundation, length):
    """Estimación de las rigideces de Winkler de una fundación de módulo
    E_foundation bajo una base de ancho 'length':

        k_normal     = E / L
        k_tangential = G / L,   G = E / (2 (1 + nu))

    Es una estimación de orden de magnitud: k no es una propiedad del
    material sino del sistema suelo-estructura.
    """
    k_normal = E_foundation / length
    k_tangential = E_foundation / (2.0 * (1.0 + nu_foundation)) / length

    return k_normal, k_tangential
