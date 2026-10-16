import numpy

from source.materials.linear_elastic import LinearElastic
from source.materials.linear_elastic import PLANE_STRAIN
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
from source.solvers.linear import solve

from tests.helpers import rectangle_mesh
from tests.test_elements import CASES, make_element, polygon_area, VERTEX_COUNT


E = 21.0e9
NU = 0.2
RHO = 2400.0
RHO_WATER = 1000.0

ELEMENT_TYPES = ["TRI3", "QUAD4", "TRI6", "QUAD9"]


def nodal_resultant(coordinates, fe):
    """Fuerza total y momento respecto del origen de un vector de fuerzas nodales."""
    Fx = 0.0
    Fy = 0.0
    Mz = 0.0

    for i in range(len(coordinates)):
        x, y = coordinates[i]
        Fx = Fx + fe[2 * i]
        Fy = Fy + fe[2 * i + 1]
        Mz = Mz + x * fe[2 * i + 1] - y * fe[2 * i]

    return Fx, Fy, Mz


def test_body_force_total_and_moment():
    # La suma de las fuerzas nodales tiene que ser el peso, y su momento el del
    # peso aplicado en el centroide.
    by = -RHO * GRAVITY

    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)
        fe = element.body_force_vector(0.0, by)

        vertices = coordinates[:VERTEX_COUNT[element_class]]
        A = polygon_area(vertices)

        # centroide del polígono
        cx = 0.0
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            cx = cx + (x1 + x2) * (x1 * y2 - x2 * y1)
        cx = cx / (6.0 * A)

        Fx, Fy, Mz = nodal_resultant(coordinates, fe)

        assert abs(Fx) < 1.0e-9
        assert abs(Fy - by * A) < 1.0e-9 * abs(by * A)
        assert abs(Mz - by * A * cx) < 1.0e-9 * abs(by * A * cx)


def test_uniform_pressure_on_quadratic_side_gives_1_4_1_distribution():
    # Presión uniforme sobre un lado cuadrático: las fuerzas nodales se reparten
    # 1/6, 4/6, 1/6 (no 1/2 - 1/2 como en el lineal).
    from tests.test_elements import QUAD_9
    from source.elements.quad9 import Quad9

    element = make_element(Quad9, QUAD_9)
    p = 1.0e5

    fe = element.side_load_vector(0, lambda x, y, n: (-p * n[0], -p * n[1]))

    start, end, middle = element.side_nodes[0]
    x0, y0 = QUAD_9[start]
    x1, y1 = QUAD_9[end]
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    total = p * L

    magnitudes = []
    for i in [start, middle, end]:
        magnitudes.append((fe[2 * i] ** 2 + fe[2 * i + 1] ** 2) ** 0.5)

    assert numpy.allclose(magnitudes, [total / 6.0, 4.0 * total / 6.0, total / 6.0])


def test_hydrostatic_side_resultant():
    # Pared vertical de altura h bajo agua hasta el borde superior:
    #   empuje F = rho_w g h^2 / 2, aplicado a h/3 de la base.
    height = 3.0
    material = LinearElastic(E, NU, RHO)
    pressure = hydrostatic_pressure(height, RHO_WATER)

    F_exact = RHO_WATER * GRAVITY * height ** 2 / 2.0
    M_exact = -F_exact * height / 3.0     # respecto del origen, empuje en +x

    for element_type in ELEMENT_TYPES:
        mesh = rectangle_mesh(1.0, height, 2, 3, element_type, material)
        elements = create_elements(mesh)

        f = numpy.zeros(mesh.number_of_dofs())
        add_side_pressure(f, mesh, elements, "left", pressure)

        Fx, Fy, Mz = nodal_resultant([[n.x, n.y] for n in mesh.nodes], f)

        assert abs(Fx - F_exact) < 1.0e-10 * F_exact
        assert abs(Fy) < 1.0e-10 * F_exact
        assert abs(Mz - M_exact) < 1.0e-10 * abs(M_exact)


def test_self_weight_column_is_exact_for_quadratic_elements():
    # Columna bajo peso propio. Solución exacta (tensión plana):
    #   syy = -rho g (H - y),  sxx = txy = 0
    #   u = nu rho g (H - y) x / E
    #   v = -rho g (H y - y^2 / 2) / E + nu rho g x^2 / (2 E)
    # Es cuadrática, así que Tri6 y Q9 la reproducen exactamente si se impone
    # en el borde.
    width = 1.0
    H = 2.0
    material = LinearElastic(E, NU, RHO)
    c = RHO * GRAVITY / E

    def exact(x, y):
        u = NU * c * (H - y) * x
        v = -c * (H * y - y * y / 2.0) + NU * c * x * x / 2.0
        return u, v

    for element_type in ["TRI6", "QUAD9"]:
        mesh = rectangle_mesh(width, H, 2, 3, element_type, material)
        elements = create_elements(mesh)

        K = DenseMatrix(mesh.number_of_dofs())
        for element in elements:
            K.scatter_local_to_global(element)

        f = numpy.zeros(mesh.number_of_dofs())
        add_self_weight(f, elements)

        K_original = K.values.copy()
        f_original = f.copy()

        for node_id in mesh.node_sets["boundary"]:
            node = mesh.node(node_id)
            dofs = mesh.node_global_dofs(node_id)
            u_exact, v_exact = exact(node.x, node.y)
            apply_dirichlet_bc(K, f, dofs[0], u_exact)
            apply_dirichlet_bc(K, f, dofs[1], v_exact)

        u = solve(K, f)

        scale = c * H * H
        for node in mesh.nodes:
            dofs = mesh.node_global_dofs(node.global_id)
            u_exact, v_exact = exact(node.x, node.y)

            assert abs(u[dofs[0]] - u_exact) < 1.0e-10 * scale
            assert abs(u[dofs[1]] - v_exact) < 1.0e-10 * scale

        # tensiones exactas en todos los puntos de Gauss
        for element in elements:
            points, weights = element.quadrature()
            for xi, eta in points:
                x, y = element.physical_coordinates(xi, eta)
                stress = element.stress(u, xi, eta)
                expected = [0.0, -RHO * GRAVITY * (H - y), 0.0]

                assert numpy.allclose(stress, expected, atol=1.0e-8 * RHO * GRAVITY * H)

        # equilibrio: las reacciones de todo el borde equilibran el peso
        R = compute_reactions(K_original, f_original, u)
        Fx, Fy, Mz = node_set_reaction_resultant(mesh, R, "boundary")
        weight = RHO * GRAVITY * width * H

        assert abs(Fx) < 1.0e-8 * weight
        assert abs(Fy - weight) < 1.0e-8 * weight


def test_reactions_balance_weight_and_water_thrust():
    # Bloque rectangular empotrado en la base, con peso propio y agua sobre la
    # cara izquierda. Las reacciones tienen que equilibrar exactamente las
    # cargas, con cualquier elemento:
    #   Rx = -F,  Ry = W,  M_reacciones(origen) = F hw/3 + W b/2
    b = 2.0
    h = 4.0
    hw = 3.0
    material = LinearElastic(E, NU, RHO)

    W = RHO * GRAVITY * b * h
    F = RHO_WATER * GRAVITY * hw ** 2 / 2.0
    M_exact = F * hw / 3.0 + W * b / 2.0

    # La presión tiene un quiebre en el nivel del agua (por encima vale cero).
    # Gauss solo integra exacto si ese quiebre cae en un nodo: con 8 filas de
    # celdas en h = 4 hay nodos cada 0.5 m, y uno de ellos está en y = hw = 3.
    for element_type in ELEMENT_TYPES:
        mesh = rectangle_mesh(b, h, 4, 8, element_type, material)
        elements = create_elements(mesh)

        K = DenseMatrix(mesh.number_of_dofs())
        for element in elements:
            K.scatter_local_to_global(element)

        f = numpy.zeros(mesh.number_of_dofs())
        add_self_weight(f, elements)
        add_side_pressure(f, mesh, elements, "left", hydrostatic_pressure(hw, RHO_WATER))

        K_original = K.values.copy()
        f_original = f.copy()

        fix_node_set(K, f, mesh, "bottom", (0, 1))

        u = solve(K, f)
        R = compute_reactions(K_original, f_original, u)

        Fx, Fy, Mz = node_set_reaction_resultant(mesh, R, "bottom", (0.0, 0.0))

        assert abs(Fx + F) < 1.0e-8 * F
        assert abs(Fy - W) < 1.0e-8 * W
        assert abs(Mz - M_exact) < 1.0e-8 * M_exact

        # en los nodos libres la reacción es nula
        free = [n for n in range(mesh.number_of_nodes()) if n not in mesh.node_sets["bottom"]]
        for node_id in free:
            for dof in mesh.node_global_dofs(node_id):
                assert abs(R[dof]) < 1.0e-6 * W


def test_self_weight_column_plane_strain_is_exact():
    # En deformación plana vale la misma solución de la columna, con las
    # constantes equivalentes E' = E/(1 - nu^2) y nu' = nu/(1 - nu).
    width = 1.0
    H = 2.0
    material = LinearElastic(E, NU, RHO, PLANE_STRAIN)
    E_eq = E / (1.0 - NU ** 2)
    nu_eq = NU / (1.0 - NU)
    c = RHO * GRAVITY / E_eq

    def exact(x, y):
        u = nu_eq * c * (H - y) * x
        v = -c * (H * y - y * y / 2.0) + nu_eq * c * x * x / 2.0
        return u, v

    for element_type in ["TRI6", "QUAD9"]:
        mesh = rectangle_mesh(width, H, 2, 3, element_type, material)
        elements = create_elements(mesh)

        K = DenseMatrix(mesh.number_of_dofs())
        for element in elements:
            K.scatter_local_to_global(element)

        f = numpy.zeros(mesh.number_of_dofs())
        add_self_weight(f, elements)

        for node_id in mesh.node_sets["boundary"]:
            node = mesh.node(node_id)
            dofs = mesh.node_global_dofs(node_id)
            u_exact, v_exact = exact(node.x, node.y)
            apply_dirichlet_bc(K, f, dofs[0], u_exact)
            apply_dirichlet_bc(K, f, dofs[1], v_exact)

        u = solve(K, f)

        scale = c * H * H
        for node in mesh.nodes:
            dofs = mesh.node_global_dofs(node.global_id)
            u_exact, v_exact = exact(node.x, node.y)

            assert abs(u[dofs[0]] - u_exact) < 1.0e-10 * scale
            assert abs(u[dofs[1]] - v_exact) < 1.0e-10 * scale
