import math
import os
import tempfile

import numpy

from source.mesh.mesh import Mesh
from source.mesh.generators import dam_mesh
from source.mesh.generators import quadrilateral_mesh
from source.mesh.generators import uniform
from source.mesh.writer import write_mesh
from source.materials.linear_elastic import LinearElastic
from source.materials.linear_elastic import PLANE_STRAIN
from source.elements.factory import create_elements
from source.systems.loads import GRAVITY
from source.systems.loads import add_side_pressure
from source.systems.loads import hydrostatic_pressure


ELEMENT_TYPES = ["TRI3", "QUAD4", "TRI6", "QUAD9"]

MATERIAL = LinearElastic(21.0e9, 0.2, 2400.0)

H = 30.0
BASE = 18.0
CREST = 3.0
FREEBOARD = 2.0
WATER = H - FREEBOARD


def total_area(elements):
    area = 0.0
    for element in elements:
        area = area + element.area()
    return area


def test_dam_area_all_element_types():
    for element_type in ELEMENT_TYPES:
        for batter in [0.0, 3.0]:
            mesh = dam_mesh(MATERIAL, element_type, 1.5, H, BASE, CREST, FREEBOARD, batter)
            elements = create_elements(mesh)

            assert abs(total_area(elements) - (BASE + CREST) / 2.0 * H) < 1.0e-9


def test_dam_corners_and_base():
    batter = 2.0
    mesh = dam_mesh(MATERIAL, "QUAD9", 1.0, H, BASE, CREST, FREEBOARD, batter)

    heel = mesh.node(mesh.node_sets["heel"][0])
    toe = mesh.node(mesh.node_sets["toe"][0])

    assert abs(heel.x + batter) < 1.0e-12 and abs(heel.y) < 1.0e-12
    assert abs(toe.x - (BASE - batter)) < 1.0e-12 and abs(toe.y) < 1.0e-12

    for node_id in mesh.node_sets["base"]:
        assert abs(mesh.node(node_id).y) < 1.0e-12

    for node_id in mesh.node_sets["crest"]:
        assert abs(mesh.node(node_id).y - H) < 1.0e-12


def test_dam_has_node_row_at_water_level():
    for size in [0.7, 1.0, 2.3]:
        mesh = dam_mesh(MATERIAL, "QUAD4", size, H, BASE, CREST, FREEBOARD)

        levels = [mesh.node(i).y for i in mesh.node_sets["upstream"]]

        assert min(abs(y - WATER) for y in levels) < 1.0e-9


def test_dam_wet_side_length():
    # el paramento mojado mide WATER (vertical) o WATER * sqrt(1 + (a/H)^2)
    for batter in [0.0, 4.0]:
        for element_type in ELEMENT_TYPES:
            mesh = dam_mesh(MATERIAL, element_type, 1.0, H, BASE, CREST, FREEBOARD, batter)
            elements = create_elements(mesh)

            length = 0.0
            for element_id, side in mesh.side_sets["upstream_wet"]:
                coordinates = elements[element_id].side_coordinates(side)
                length = length + numpy.linalg.norm(coordinates[1] - coordinates[0])

            expected = WATER * math.sqrt(1.0 + (batter / H) ** 2)
            assert abs(length - expected) < 1.0e-9


def test_dam_water_thrust_is_exact():
    # Empuje horizontal sobre el paramento vertical: F = rho_w g hw^2 / 2,
    # aplicado a hw/3 de la base.
    F = 1000.0 * GRAVITY * WATER ** 2 / 2.0

    for element_type in ELEMENT_TYPES:
        mesh = dam_mesh(MATERIAL, element_type, 1.3, H, BASE, CREST, FREEBOARD)
        elements = create_elements(mesh)

        f = numpy.zeros(mesh.number_of_dofs())
        add_side_pressure(f, mesh, elements, "upstream_wet", hydrostatic_pressure(WATER))

        Fx = f[0::2].sum()
        Fy = f[1::2].sum()
        M = 0.0
        for node in mesh.nodes:
            dofs = mesh.node_global_dofs(node.global_id)
            M = M + node.x * f[dofs[1]] - node.y * f[dofs[0]]

        assert abs(Fx - F) < 1.0e-9 * F
        assert abs(Fy) < 1.0e-9 * F
        assert abs(M + F * WATER / 3.0) < 1.0e-9 * F * WATER


def test_quadratic_center_node_is_bilinear_center():
    corners = [(0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (0.5, 3.0)]
    mesh = quadrilateral_mesh(corners, uniform(1), uniform(1), "QUAD9", {1: MATERIAL})

    center = mesh.node(mesh.connectivity[0][8])

    assert abs(center.x - sum(c[0] for c in corners) / 4.0) < 1.0e-12
    assert abs(center.y - sum(c[1] for c in corners) / 4.0) < 1.0e-12


def test_write_and_read_back():
    material = LinearElastic(21.0e9, 0.2, 2400.0, PLANE_STRAIN)

    for element_type in ELEMENT_TYPES:
        mesh = dam_mesh(material, element_type, 2.0, H, BASE, CREST, FREEBOARD, 1.0, 0.5)

        handle, filename = tempfile.mkstemp(suffix=".mesh")
        os.close(handle)

        try:
            write_mesh(filename, mesh, "test\nde dos líneas")
            read = Mesh()
            read.read(filename)
        finally:
            os.remove(filename)

        assert read.number_of_nodes() == mesh.number_of_nodes()
        assert read.connectivity == mesh.connectivity
        assert read.element_types == mesh.element_types
        assert read.thickness == 0.5
        assert read.node_sets == mesh.node_sets
        assert read.side_sets == mesh.side_sets

        for a, b in zip(read.nodes, mesh.nodes):
            assert abs(a.x - b.x) < 1.0e-9 and abs(a.y - b.y) < 1.0e-9

        m = read.materials[1]
        assert m.E == 21.0e9 and m.nu == 0.2 and m.density == 2400.0
        assert m.hypothesis == PLANE_STRAIN


def test_invalid_dam_parameters():
    for kwargs in [{"freeboard": 0.0}, {"freeboard": 40.0}, {"crest": 20.0}]:
        try:
            dam_mesh(MATERIAL, **kwargs)
        except ValueError:
            continue

        assert False, "Expected ValueError for " + str(kwargs)
