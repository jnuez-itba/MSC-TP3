"""Ensamblado del vector global de cargas f.

Cada elemento calcula su vector de fuerzas nodales equivalentes fe y acá se
suma en las posiciones de sus grados de libertad, igual que el ensamblado de
K en DenseMatrix.scatter_local_to_global.
"""

GRAVITY = 9.81


def scatter_vector(f, element, fe):
    dofs = element.global_dofs()

    for i in range(len(dofs)):
        f[dofs[i]] = f[dofs[i]] + fe[i]


def add_self_weight(f, elements, gravity=GRAVITY):
    """Peso propio: fuerza de volumen b = (0, -rho g) en cada elemento, con la
    densidad de su material."""
    for element in elements:
        by = -element.material.density * gravity

        if by != 0.0:
            scatter_vector(f, element, element.body_force_vector(0.0, by))


def add_side_traction(f, mesh, elements, side_set_name, traction_function):
    """Tracción sobre los lados de un SideSet.

    traction_function(x, y, normal) -> (tx, ty) en N/m2.
    """
    if side_set_name not in mesh.side_sets:
        raise ValueError("Unknown SideSet: " + side_set_name)

    for element_id, side in mesh.side_sets[side_set_name]:
        element = elements[element_id]
        scatter_vector(f, element, element.side_load_vector(side, traction_function))


def add_side_pressure(f, mesh, elements, side_set_name, pressure_function):
    """Presión sobre los lados de un SideSet.

    pressure_function(x, y) -> p en N/m2, positiva en compresión (empuja hacia
    adentro del sólido). Es una tracción t = -p n, con n la normal saliente.
    """
    def traction(x, y, normal):
        p = pressure_function(x, y)
        return -p * normal[0], -p * normal[1]

    add_side_traction(f, mesh, elements, side_set_name, traction)


def hydrostatic_pressure(water_level, water_density=1000.0, gravity=GRAVITY):
    """Presión hidrostática p = rho_w g (nivel - y) por debajo del nivel del
    agua, y cero por encima."""
    def pressure(x, y):
        depth = water_level - y

        if depth <= 0.0:
            return 0.0

        return water_density * gravity * depth

    return pressure
