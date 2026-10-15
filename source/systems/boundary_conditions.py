def apply_dirichlet_bc(K, f, dof, value):
    for i in range(K.number_of_dofs):
        f[i] = f[i] - K.values[i][dof] * value

    for i in range(K.number_of_dofs):
        K.values[dof][i] = 0.0
        K.values[i][dof] = 0.0

    K.values[dof][dof] = 1.0
    f[dof] = value


def fix_node_set(K, f, mesh, node_set_name, components=(0, 1), value=0.0):
    """Impone un desplazamiento en todos los nodos de un NodeSet.

    components  (0, 1) fija u y v (empotramiento)
                (0,)   fija solo u
                (1,)   fija solo v (apoyo deslizante horizontal)
    """
    if node_set_name not in mesh.node_sets:
        raise ValueError("Unknown NodeSet: " + node_set_name)

    for node_id in mesh.node_sets[node_set_name]:
        dofs = mesh.node_global_dofs(node_id)

        for component in components:
            apply_dirichlet_bc(K, f, dofs[component], value)
