"""Post-proceso: desplazamientos y tensiones nodales, tensiones principales."""

import numpy


def nodal_displacements(mesh, u):
    """Matriz n_nodos x 2 con [u, v] de cada nodo."""
    displacements = numpy.zeros((mesh.number_of_nodes(), 2))

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        displacements[node.global_id][0] = u[dofs[0]]
        displacements[node.global_id][1] = u[dofs[1]]

    return displacements


def nodal_stresses(mesh, elements, u):
    """Matriz n_nodos x 3 con [sxx, syy, txy] en cada nodo.

    La tensión de elementos finitos es discontinua entre elementos: cada
    elemento da un valor distinto en un nodo compartido. Se toma el promedio
    de los elementos que comparten el nodo (suavizado nodal).
    """
    n_nodes = mesh.number_of_nodes()

    total = numpy.zeros((n_nodes, 3))
    count = numpy.zeros(n_nodes)

    for element in elements:
        stresses = element.stress_at_nodes(u)

        for a in range(element.number_of_nodes):
            node_id = element.node_global_ids[a]
            total[node_id] = total[node_id] + stresses[a]
            count[node_id] = count[node_id] + 1

    for i in range(n_nodes):
        if count[i] > 0:
            total[i] = total[i] / count[i]

    return total


def principal_stresses(sxx, syy, txy):
    """Tensiones principales en el plano.

        s1, s2 = (sxx + syy)/2 +- sqrt( ((sxx - syy)/2)^2 + txy^2 )
        theta  = 1/2 atan2(2 txy, sxx - syy)   (ángulo de s1 con el eje x)

    Funciona con números o con arrays de numpy. s1 >= s2.
    """
    center = (sxx + syy) / 2.0
    radius = numpy.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)
    theta = 0.5 * numpy.arctan2(2.0 * txy, sxx - syy)

    return center + radius, center - radius, theta


def von_mises(sxx, syy, txy):
    """Tensión equivalente de von Mises en tensión plana (szz = 0)."""
    return numpy.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy)


def stress_fields(mesh, elements, u):
    """Diccionario con todos los campos nodales de tensión, en Pa."""
    stresses = nodal_stresses(mesh, elements, u)

    sxx = stresses[:, 0]
    syy = stresses[:, 1]
    txy = stresses[:, 2]

    s1, s2, theta = principal_stresses(sxx, syy, txy)

    return {
        "sigma_xx": sxx,
        "sigma_yy": syy,
        "tau_xy": txy,
        "sigma_1": s1,
        "sigma_2": s2,
        "von_mises": von_mises(sxx, syy, txy),
    }
