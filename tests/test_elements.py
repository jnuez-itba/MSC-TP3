import numpy

from source.mesh.mesh import Mesh
from source.mesh.node import Node
from source.materials.linear_elastic import LinearElastic
from source.elements.tri3 import Tri3
from source.elements.quad4 import Quad4
from source.elements.tri6 import Tri6
from source.elements.quad9 import Quad9


E = 21.0e9
NU = 0.2
THICKNESS = 1.0


def make_mesh(coordinates):
    mesh = Mesh()
    mesh.dofs_per_node = 2

    for i in range(len(coordinates)):
        mesh.nodes.append(Node(i, coordinates[i][0], coordinates[i][1]))

    return mesh


def make_element(element_class, coordinates):
    mesh = make_mesh(coordinates)
    material = LinearElastic(E, NU)
    node_ids = list(range(len(coordinates)))

    return element_class(0, node_ids, mesh, material, THICKNESS)


def midpoint(p, q):
    return [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0]


# Geometrías de prueba (vértices en sentido antihorario)
TRIANGLE = [[0.0, 0.0], [2.0, 0.5], [0.5, 1.5]]
QUAD_DISTORTED = [[0.0, 0.0], [2.0, 0.2], [2.3, 1.7], [0.1, 1.2]]

# Elementos cuadráticos de lados rectos: nodos de lado en el punto medio y,
# para el Q9, nodo central en el centro de la cuadrilátero bilineal
TRIANGLE_6 = TRIANGLE + [
    midpoint(TRIANGLE[0], TRIANGLE[1]),
    midpoint(TRIANGLE[1], TRIANGLE[2]),
    midpoint(TRIANGLE[2], TRIANGLE[0]),
]

QUAD_9 = QUAD_DISTORTED + [
    midpoint(QUAD_DISTORTED[0], QUAD_DISTORTED[1]),
    midpoint(QUAD_DISTORTED[1], QUAD_DISTORTED[2]),
    midpoint(QUAD_DISTORTED[2], QUAD_DISTORTED[3]),
    midpoint(QUAD_DISTORTED[3], QUAD_DISTORTED[0]),
    [
        sum(p[0] for p in QUAD_DISTORTED) / 4.0,
        sum(p[1] for p in QUAD_DISTORTED) / 4.0,
    ],
]

CASES = [
    (Tri3, TRIANGLE),
    (Quad4, QUAD_DISTORTED),
    (Tri6, TRIANGLE_6),
    (Quad9, QUAD_9),
]

# Coordenadas naturales de los nodos de cada elemento
NATURAL_NODES = {
    Tri3: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    Quad4: [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    Tri6: [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
    ],
    Quad9: [
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        [0.0, 0.0],
    ],
}

# Puntos interiores de prueba en coordenadas naturales
SAMPLE_POINTS = {
    Tri3: [[0.1, 0.2], [0.3, 0.5], [0.6, 0.1]],
    Tri6: [[0.1, 0.2], [0.3, 0.5], [0.6, 0.1]],
    Quad4: [[0.1, 0.2], [-0.4, 0.7], [0.9, -0.3]],
    Quad9: [[0.1, 0.2], [-0.4, 0.7], [0.9, -0.3]],
}

VERTEX_COUNT = {Tri3: 3, Tri6: 3, Quad4: 4, Quad9: 4}


def polygon_area(coordinates):
    # fórmula del área de Gauss (shoelace)
    A = 0.0
    n = len(coordinates)

    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        A = A + x1 * y2 - x2 * y1

    return A / 2.0


def test_partition_of_unity():
    # sum(Ni) = 1 y sum(dNi) = 0 en cualquier punto
    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)

        for xi, eta in SAMPLE_POINTS[element_class]:
            N = element.shape_functions(xi, eta)
            dN = element.shape_derivatives(xi, eta)

            assert abs(N.sum() - 1.0) < 1.0e-14
            assert numpy.allclose(dN.sum(axis=1), 0.0)


def test_shape_functions_are_one_at_own_node():
    # Ni(nodo j) = 1 si i == j, 0 si no (propiedad de Kronecker)
    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)
        nodes = NATURAL_NODES[element_class]

        for j in range(len(nodes)):
            N = element.shape_functions(nodes[j][0], nodes[j][1])
            expected = numpy.zeros(len(nodes))
            expected[j] = 1.0

            assert numpy.allclose(N, expected)


def test_shape_derivatives_match_finite_differences():
    h = 1.0e-6

    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)

        for xi, eta in SAMPLE_POINTS[element_class]:
            dN = element.shape_derivatives(xi, eta)

            dN_dxi = (element.shape_functions(xi + h, eta)
                      - element.shape_functions(xi - h, eta)) / (2.0 * h)
            dN_deta = (element.shape_functions(xi, eta + h)
                       - element.shape_functions(xi, eta - h)) / (2.0 * h)

            assert numpy.allclose(dN[0], dN_dxi, atol=1.0e-8)
            assert numpy.allclose(dN[1], dN_deta, atol=1.0e-8)


def test_area():
    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)
        vertices = coordinates[:VERTEX_COUNT[element_class]]

        assert abs(element.area() - polygon_area(vertices)) < 1.0e-12


def test_stiffness_is_symmetric():
    for element_class, coordinates in CASES:
        Ke = make_element(element_class, coordinates).stiffness_matrix()

        assert numpy.allclose(Ke, Ke.transpose(), rtol=1.0e-12, atol=1.0e-3)


def test_stiffness_has_three_rigid_body_modes():
    # Un elemento libre tiene exactamente 3 modos de energía nula:
    # 2 traslaciones y 1 rotación. El resto de los autovalores es positivo.
    # (Si la integración fuera reducida aparecerían modos espurios extra.)
    for element_class, coordinates in CASES:
        Ke = make_element(element_class, coordinates).stiffness_matrix()

        eigenvalues = numpy.linalg.eigvalsh(Ke)
        tolerance = 1.0e-8 * eigenvalues.max()

        n_zero = 0
        for value in eigenvalues:
            assert value > -tolerance
            if abs(value) < tolerance:
                n_zero = n_zero + 1

        assert n_zero == 3


def test_rigid_body_motion_produces_no_stress():
    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)

        # traslación (0.3, -0.2) + rotación infinitesimal theta = 1e-3
        theta = 1.0e-3
        u = []
        for x, y in coordinates:
            u.append(0.3 - theta * y)
            u.append(-0.2 + theta * x)

        u = numpy.array(u)

        assert numpy.allclose(element.Ke @ u, 0.0, atol=1.0e-6 * E)

        for stress in element.stress_at_gauss_points(u):
            assert numpy.allclose(stress, 0.0, atol=1.0e-6 * E * 1.0e-3)


def test_linear_displacement_gives_exact_constant_stress():
    # u = a1 + a2 x + a3 y, v = b1 + b2 x + b3 y
    # => exx = a2, eyy = b3, gxy = a3 + b2 (constante)
    a = [1.0e-4, 2.0e-4, -1.0e-4]
    b = [-2.0e-4, 0.5e-4, -3.0e-4]

    strain_exact = numpy.array([a[1], b[2], a[2] + b[1]])
    material = LinearElastic(E, NU)
    stress_exact = material.D @ strain_exact

    for element_class, coordinates in CASES:
        element = make_element(element_class, coordinates)

        u = []
        for x, y in coordinates:
            u.append(a[0] + a[1] * x + a[2] * y)
            u.append(b[0] + b[1] * x + b[2] * y)

        u = numpy.array(u)

        for stress in element.stress_at_gauss_points(u):
            assert numpy.allclose(stress, stress_exact, rtol=1.0e-10)


def test_quadratic_elements_reproduce_quadratic_field():
    # Los elementos cuadráticos representan exactamente un campo cuadrático
    # completo: la deformación (lineal) tiene que salir exacta en cualquier
    # punto. Los lineales no pueden, y eso es lo que los hace más precisos.
    #   u = x^2 + 2xy - y^2,  v = 3x^2 - xy + 2y^2   (escalado por 1e-4)
    c = 1.0e-4

    def field(x, y):
        return (c * (x * x + 2.0 * x * y - y * y),
                c * (3.0 * x * x - x * y + 2.0 * y * y))

    def strain(x, y):
        exx = c * (2.0 * x + 2.0 * y)
        eyy = c * (-x + 4.0 * y)
        gxy = c * (2.0 * x - 2.0 * y) + c * (6.0 * x - y)
        return numpy.array([exx, eyy, gxy])

    for element_class, coordinates in [(Tri6, TRIANGLE_6), (Quad9, QUAD_9)]:
        element = make_element(element_class, coordinates)

        u = []
        for x, y in coordinates:
            ux, uy = field(x, y)
            u.append(ux)
            u.append(uy)

        u = numpy.array(u)

        for xi, eta in SAMPLE_POINTS[element_class]:
            x, y = element.physical_coordinates(xi, eta)

            assert numpy.allclose(
                element.strain(u, xi, eta), strain(x, y), rtol=1.0e-10, atol=1.0e-16
            )


def test_tri3_matches_closed_form_cst():
    # Para el CST: Ke = t A B^T D B con B constante,
    # B = 1/(2A) [[b1 0 b2 0 b3 0], [0 c1 0 c2 0 c3], [c1 b1 c2 b2 c3 b3]]
    # bi = yj - yk, ci = xk - xj (i, j, k cíclicos)
    (x1, y1), (x2, y2), (x3, y3) = TRIANGLE
    A = polygon_area(TRIANGLE)

    b = [y2 - y3, y3 - y1, y1 - y2]
    c = [x3 - x2, x1 - x3, x2 - x1]

    B = numpy.zeros((3, 6))
    for i in range(3):
        B[0][2 * i] = b[i]
        B[1][2 * i + 1] = c[i]
        B[2][2 * i] = c[i]
        B[2][2 * i + 1] = b[i]
    B = B / (2.0 * A)

    material = LinearElastic(E, NU)
    Ke_exact = THICKNESS * A * B.transpose() @ material.D @ B

    element = make_element(Tri3, TRIANGLE)

    assert numpy.allclose(element.stiffness_matrix(), Ke_exact, rtol=1.0e-12)


def test_clockwise_element_is_rejected():
    clockwise = [QUAD_DISTORTED[0], QUAD_DISTORTED[3], QUAD_DISTORTED[2], QUAD_DISTORTED[1]]

    try:
        make_element(Quad4, clockwise)
    except ValueError:
        return

    assert False, "Expected ValueError for clockwise node order"
