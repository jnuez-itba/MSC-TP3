import os

import numpy

from source.mesh.mesh import Mesh
from source.mesh.generators import dam_mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.factory import create_elements
from source.systems.loads import GRAVITY
from source.dam.geometry import DamGeometry
from source.dam.stability import stability
from source.dam.stability import trapezoidal_base_pressure
from source.dam.model import solve_dam
from source.output.postprocess import find_element
from source.output.postprocess import stress_at_point
from source.verification.patch import exact_displacement
from source.verification.patch import exact_strain


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATERIAL = LinearElastic(21.0e9, 0.2, 2400.0)


def test_hand_calculation_default_dam():
    # Valores calculados a mano para la presa del enunciado
    g = GRAVITY
    result = stability(DamGeometry())

    W = 2400.0 * g * 315.0
    F = 1000.0 * g * 28.0 ** 2 / 2.0
    Ms = 2400.0 * g * (3.0 * 30.0 * 16.5 + 15.0 * 30.0 / 2.0 * 10.0)
    Md = F * 28.0 / 3.0

    assert abs(result["weight"] - W) < 1.0e-6 * W
    assert abs(result["water_horizontal"] - F) < 1.0e-6 * F
    assert abs(result["moment_stabilizing"] - Ms) < 1.0e-6 * Ms
    assert abs(result["moment_overturning"] - Md) < 1.0e-6 * Md
    assert abs(result["safety_factor"] - Ms / Md) < 1.0e-9
    assert result["middle_third"]
    assert abs(result["resultant_from_toe"] - (Ms - Md) / W) < 1.0e-9


def test_trapezoidal_pressure_resultant():
    # La ley trapecial tiene que dar la fuerza N y el momento respecto del pie
    geometry = DamGeometry(upstream_batter=2.0)
    result = stability(geometry)

    x_heel = geometry.heel()[0]
    x_toe = geometry.toe()[0]
    x = numpy.linspace(x_heel, x_toe, 2001)
    q = numpy.array([trapezoidal_base_pressure(geometry, result, xi) for xi in x])

    N = numpy.trapezoid(q, x) if hasattr(numpy, "trapezoid") else numpy.trapz(q, x)
    M = (numpy.trapezoid(q * (x_toe - x), x) if hasattr(numpy, "trapezoid")
         else numpy.trapz(q * (x_toe - x), x))

    assert abs(N - result["normal_force"]) < 1.0e-6 * result["normal_force"]
    assert abs(M - result["normal_force"] * result["resultant_from_toe"]) < 1.0e-6 * M


def check_reactions_match_hand(geometry, element_type, support, size=3.0):
    hand = stability(geometry)

    mesh = dam_mesh(MATERIAL, element_type, size, geometry.height, geometry.base,
                    geometry.crest, geometry.freeboard, geometry.upstream_batter)
    result = solve_dam(mesh, geometry.water_level(), support)

    net = hand["moment_stabilizing"] - hand["moment_overturning"]

    assert abs(result["reaction_x"] + hand["water_horizontal"]) < 1.0e-8 * hand["water_horizontal"]
    assert abs(result["reaction_y"] - hand["normal_force"]) < 1.0e-8 * hand["normal_force"]
    assert abs(result["reaction_moment_toe"] + net) < 1.0e-8 * net

    return result


def test_fem_reactions_match_hand_calculation():
    # El momento de las reacciones respecto del pie es igual y opuesto al
    # momento neto de las cargas, para cualquier elemento y apoyo
    for element_type in ["TRI3", "QUAD4", "TRI6", "QUAD9"]:
        for support in ["fixed", "roller"]:
            check_reactions_match_hand(DamGeometry(), element_type, support)


def test_fem_reactions_with_upstream_batter():
    # Con talud aguas arriba el agua también empuja hacia abajo (peso de la
    # cuña de agua sobre el paramento). El cálculo manual y el FEM tienen que
    # coincidir.
    geometry = DamGeometry(upstream_batter=4.0)

    assert stability(geometry)["water_vertical"] > 0.0

    for element_type in ["QUAD4", "QUAD9"]:
        check_reactions_match_hand(geometry, element_type, "fixed")


def test_roller_support_takes_thrust_at_heel():
    geometry = DamGeometry()
    result = check_reactions_match_hand(geometry, "QUAD9", "roller")

    mesh = result["mesh"]
    heel = mesh.node_sets["heel"][0]
    Rx_heel = result["reactions"][mesh.node_global_dofs(heel)[0]]

    F = stability(geometry)["water_horizontal"]
    assert abs(Rx_heel + F) < 1.0e-8 * F


def test_stress_at_point_is_exact_for_linear_field():
    for name in ["tri3", "quad4", "tri6", "quad9"]:
        mesh = Mesh()
        mesh.read(os.path.join(ROOT, "meshes", "patch", "patch_" + name + ".mesh"))
        mesh.dofs_per_node = 2
        elements = create_elements(mesh)

        u = numpy.zeros(mesh.number_of_dofs())
        for node in mesh.nodes:
            dofs = mesh.node_global_dofs(node.global_id)
            u[dofs[0]], u[dofs[1]] = exact_displacement(node.x, node.y)

        expected = elements[0].material.D @ exact_strain()

        for x, y in [(0.01, 0.01), (0.12, 0.06), (0.2, 0.1), (0.24, 0.12), (0.07, 0.03)]:
            assert numpy.allclose(stress_at_point(elements, u, x, y), expected, rtol=1.0e-9)


def test_find_element_inverse_mapping():
    mesh = dam_mesh(MATERIAL, "QUAD9", 2.0)
    elements = create_elements(mesh)

    for x, y in [(0.5, 0.5), (9.0, 0.0), (2.0, 20.0), (1.0, 29.9)]:
        element, natural = find_element(elements, x, y)
        assert element is not None
        assert numpy.allclose(element.physical_coordinates(natural[0], natural[1]), [x, y])

    # fuera de la presa (aguas arriba y por encima del talud)
    for x, y in [(-1.0, 5.0), (17.0, 20.0)]:
        element, natural = find_element(elements, x, y)
        assert element is None
