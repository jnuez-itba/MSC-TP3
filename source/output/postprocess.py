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


def nodal_stress_components(mesh, elements, u):
    """Matriz n_nodos x 4 con [sxx, syy, txy, szz] en cada nodo.

    La tensión de elementos finitos es discontinua entre elementos: cada
    elemento da un valor distinto en un nodo compartido. Se toma el promedio
    de los elementos que comparten el nodo (suavizado nodal).

    szz es la tensión fuera del plano: 0 en tensión plana y nu (sxx + syy)
    en deformación plana (la calcula el material de cada elemento).
    """
    n_nodes = mesh.number_of_nodes()

    total = numpy.zeros((n_nodes, 4))
    count = numpy.zeros(n_nodes)

    for element in elements:
        stresses = element.stress_at_nodes(u)

        for a in range(element.number_of_nodes):
            node_id = element.node_global_ids[a]
            sxx, syy, txy = stresses[a]
            szz = element.material.out_of_plane_stress(sxx, syy)

            total[node_id] = total[node_id] + numpy.array([sxx, syy, txy, szz])
            count[node_id] = count[node_id] + 1

    for i in range(n_nodes):
        if count[i] > 0:
            total[i] = total[i] / count[i]

    return total


def nodal_stresses(mesh, elements, u):
    """Matriz n_nodos x 3 con [sxx, syy, txy] en cada nodo (promediadas)."""
    return nodal_stress_components(mesh, elements, u)[:, 0:3]


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


def von_mises(sxx, syy, txy, szz=0.0):
    """Tensión equivalente de von Mises con la tensión fuera del plano szz
    (szz = 0 en tensión plana)."""
    return numpy.sqrt(sxx * sxx + syy * syy + szz * szz
                      - sxx * syy - syy * szz - szz * sxx + 3.0 * txy * txy)


def stress_fields(mesh, elements, u):
    """Diccionario con todos los campos nodales de tensión, en Pa.

    sigma_1, sigma_2  tensiones principales EN EL PLANO de la sección
    sigma_zz          tensión fuera del plano (no nula en deformación plana)
    von_mises         tensión equivalente, incluyendo sigma_zz
    """
    stresses = nodal_stress_components(mesh, elements, u)

    sxx = stresses[:, 0]
    syy = stresses[:, 1]
    txy = stresses[:, 2]
    szz = stresses[:, 3]

    s1, s2, theta = principal_stresses(sxx, syy, txy)

    return {
        "sigma_xx": sxx,
        "sigma_yy": syy,
        "tau_xy": txy,
        "sigma_zz": szz,
        "sigma_1": s1,
        "sigma_2": s2,
        "von_mises": von_mises(sxx, syy, txy, szz),
    }


def find_element(elements, x, y):
    """Elemento que contiene el punto (x, y) y sus coordenadas naturales.

    Devuelve (elemento, (xi, eta)) o (None, None) si el punto está fuera de
    la malla. Si el punto está sobre un lado compartido, devuelve el primer
    elemento que lo contiene.
    """
    for element in elements:
        x_min, y_min = element.coordinates.min(axis=0)
        x_max, y_max = element.coordinates.max(axis=0)
        margin = 1.0e-9 * max(x_max - x_min, y_max - y_min)

        if (x < x_min - margin or x > x_max + margin
                or y < y_min - margin or y > y_max + margin):
            continue

        natural = element.natural_coordinates(x, y)

        if natural is not None and element.contains_natural(natural[0], natural[1]):
            return element, natural

    return None, None


def stress_at_point(elements, u, x, y):
    """Tensión [sxx, syy, txy] en un punto cualquiera de la malla, evaluada
    con las funciones de forma del elemento que lo contiene (sin promediar).
    Sirve para comparar mallas distintas en un mismo punto."""
    element, natural = find_element(elements, x, y)

    if element is None:
        raise ValueError("Point ({}, {}) is outside the mesh".format(x, y))

    return element.stress(u, natural[0], natural[1])
