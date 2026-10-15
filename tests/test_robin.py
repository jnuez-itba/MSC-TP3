import numpy

from source.mesh.generators import dam_mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.quad4 import Quad4
from source.elements.quad9 import Quad9
from source.elements.tri6 import Tri6
from source.dam.geometry import DamGeometry
from source.dam.stability import stability
from source.dam.stability import trapezoidal_base_pressure
from source.dam.model import solve_dam
from source.dam.model import base_nodes_sorted
from source.systems.robin import winkler_stiffness
from source.output.postprocess import nodal_displacements

from tests.test_elements import make_element, QUAD_DISTORTED, QUAD_9, TRIANGLE_6


K_N = 3.0e9
K_T = 1.0e9


def side_length(element, side):
    coordinates = element.side_coordinates(side)
    return numpy.linalg.norm(coordinates[1] - coordinates[0])


def side_vectors(element, side):
    coordinates = element.side_coordinates(side)
    tangent = (coordinates[1] - coordinates[0]) / numpy.linalg.norm(coordinates[1] - coordinates[0])
    normal = numpy.array([tangent[1], -tangent[0]])
    return normal, tangent


def test_spring_matrix_rigid_translation():
    # Si todo el lado se traslada un desplazamiento d, la fuerza total de los
    # resortes es -(k_n (d.n) n + k_t (d.s) s) L t. Lados rectos lineales y
    # cuadráticos, en distintas orientaciones.
    cases = [(Quad4, QUAD_DISTORTED), (Quad9, QUAD_9), (Tri6, TRIANGLE_6)]

    for element_class, coordinates in cases:
        element = make_element(element_class, coordinates)

        for side in range(len(element.side_nodes)):
            Ks = element.side_spring_matrix(side, K_N, K_T)
            L = side_length(element, side)
            normal, tangent = side_vectors(element, side)

            for d in [normal, tangent, numpy.array([0.3, -0.7])]:
                u = numpy.tile(d, element.number_of_nodes)
                force = -(Ks @ u)
                total = numpy.array([force[0::2].sum(), force[1::2].sum()])

                expected = -(K_N * (d @ normal) * normal
                             + K_T * (d @ tangent) * tangent) * L * element.thickness

                assert numpy.allclose(total, expected, rtol=1.0e-10, atol=1.0e-3)


def test_spring_matrix_symmetric_positive():
    element = make_element(Quad9, QUAD_9)
    Ks = element.side_spring_matrix(1, K_N, K_T)

    assert numpy.allclose(Ks, Ks.transpose())
    assert numpy.linalg.eigvalsh(Ks).min() > -1.0e-6 * K_N


def test_robin_reactions_match_hand_calculation():
    geometry = DamGeometry()
    hand = stability(geometry)
    k_n, k_t = winkler_stiffness(20.0e9, 0.25, geometry.base)

    for element_type in ["QUAD4", "QUAD9", "TRI6"]:
        mesh = dam_mesh(LinearElastic(21.0e9, 0.2, 2400.0), element_type, 3.0)
        result = solve_dam(mesh, geometry.water_level(), "robin", k_normal=k_n, k_tangential=k_t)

        net = hand["moment_stabilizing"] - hand["moment_overturning"]

        assert abs(result["reaction_x"] + hand["water_horizontal"]) < 1.0e-8 * hand["water_horizontal"]
        assert abs(result["reaction_y"] - hand["weight"]) < 1.0e-8 * hand["weight"]
        assert abs(result["reaction_moment_toe"] + net) < 1.0e-8 * net


def crest_displacement(result):
    mesh = result["mesh"]
    d = nodal_displacements(mesh, result["u"])
    return max(d[i][0] for i in mesh.node_sets["crest"])


def test_stiff_springs_recover_fixed_base():
    material = LinearElastic(21.0e9, 0.2, 2400.0)

    fixed = solve_dam(dam_mesh(material, "QUAD9", 2.0), 28.0, "fixed")
    robin = solve_dam(dam_mesh(material, "QUAD9", 2.0), 28.0, "robin",
                      k_normal=1.0e14, k_tangential=1.0e14)

    assert abs(crest_displacement(robin) - crest_displacement(fixed)) < 1.0e-3 * crest_displacement(fixed)


def test_rigid_dam_on_springs_gives_trapezoidal_law():
    # Si la presa es mucho más rígida que la fundación, se mueve como cuerpo
    # rígido sobre los resortes: el hundimiento es lineal en x y la presión
    # de contacto p = k_n * hundimiento es exactamente la ley trapecial.
    geometry = DamGeometry()
    hand = stability(geometry)

    rigid = LinearElastic(21.0e15, 0.2, 2400.0)
    k_n, k_t = winkler_stiffness(1.0e9, 0.25, geometry.base)

    mesh = dam_mesh(rigid, "QUAD9", 2.0)
    result = solve_dam(mesh, geometry.water_level(), "robin", k_normal=k_n, k_tangential=k_t)

    d = nodal_displacements(mesh, result["u"])

    for node_id in base_nodes_sorted(mesh):
        x = mesh.node(node_id).x
        pressure = -k_n * d[node_id][1]
        expected = trapezoidal_base_pressure(geometry, hand, x)

        assert abs(pressure - expected) < 1.0e-3 * hand["base_pressure_toe"]
