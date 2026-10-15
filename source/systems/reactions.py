"""Reacciones de vínculo.

apply_dirichlet_bc modifica K y f, así que hay que guardar una copia ANTES de
aplicar los apoyos:

    K_original = K.values.copy()
    f_original = f.copy()
    ... aplicar apoyos, resolver u ...
    R = compute_reactions(K_original, f_original, u)

En los grados de libertad libres R es (numéricamente) cero; en los fijos es la
fuerza que hace el apoyo sobre la estructura.
"""


def compute_reactions(K_original, f_original, u):
    """R = K u - f, con K y f sin las condiciones de borde."""
    return K_original @ u - f_original


def reaction_resultant(mesh, R, node_ids, point=(0.0, 0.0)):
    """Resultante de las reacciones en un grupo de nodos.

    Devuelve (Fx, Fy, Mz): fuerzas por unidad de espesor ya incluidas en R y
    momento respecto de 'point', positivo antihorario:

        Mz = sum( (x - x0) Ry - (y - y0) Rx )
    """
    Fx = 0.0
    Fy = 0.0
    Mz = 0.0

    for node_id in node_ids:
        node = mesh.node(node_id)
        dofs = mesh.node_global_dofs(node_id)

        Rx = R[dofs[0]]
        Ry = R[dofs[1]]

        Fx = Fx + Rx
        Fy = Fy + Ry
        Mz = Mz + (node.x - point[0]) * Ry - (node.y - point[1]) * Rx

    return Fx, Fy, Mz


def node_set_reaction_resultant(mesh, R, node_set_name, point=(0.0, 0.0)):
    if node_set_name not in mesh.node_sets:
        raise ValueError("Unknown NodeSet: " + node_set_name)

    return reaction_resultant(mesh, R, mesh.node_sets[node_set_name], point)
