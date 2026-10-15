import os
import tempfile

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import PLANE_STRESS
from source.elements.factory import create_elements


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VALID_MESH = """
# malla de prueba: 2 triángulos lineales
$Parameters
thickness 0.5
$EndParameters

$Materials
1 ELASTIC_PLANE_STRESS 21.0e9 0.2 2400.0
$EndMaterials

$Nodes
4
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
$EndNodes

$Elements
2
1 TRI 1 1   1 2 3
2 TRI 1 1   1 3 4
$EndElements

$NodeSets
left 2
1 4
bottom 2 1 2
$EndNodeSets

$SideSets
top 1 2 2
$EndSideSets
"""


def read_text(text):
    handle, filename = tempfile.mkstemp(suffix=".mesh")
    os.close(handle)

    file = open(filename, "w")
    file.write(text)
    file.close()

    try:
        mesh = Mesh()
        mesh.read(filename)
    finally:
        os.remove(filename)

    return mesh


def expect_error(text):
    try:
        read_text(text)
    except ValueError:
        return

    assert False, "Expected ValueError"


def test_read_sections():
    mesh = read_text(VALID_MESH)

    assert mesh.number_of_nodes() == 4
    assert mesh.number_of_elements() == 2
    assert mesh.thickness == 0.5

    assert mesh.element_types == ["TRI3", "TRI3"]
    assert mesh.element_families == ["TRI", "TRI"]
    assert mesh.element_orders == [1, 1]
    assert mesh.connectivity[1] == [0, 2, 3]

    material = mesh.element_material(0)
    assert material.E == 21.0e9
    assert material.nu == 0.2
    assert material.density == 2400.0
    assert material.hypothesis == PLANE_STRESS

    # los conjuntos pueden ocupar una o varias líneas
    assert mesh.node_sets["left"] == [0, 3]
    assert mesh.node_sets["bottom"] == [0, 1]

    # lado 2 del elemento 2 (numeración interna desde 0)
    assert mesh.side_sets["top"] == [(1, 1)]


def test_elements_get_material_and_thickness_from_file():
    mesh = read_text(VALID_MESH)
    mesh.dofs_per_node = 2

    elements = create_elements(mesh)

    assert len(elements) == 2
    assert elements[0].thickness == 0.5
    assert elements[0].material.E == 21.0e9


def test_example_file():
    mesh = Mesh()
    mesh.read(os.path.join(ROOT, "meshes", "ejemplo_formato.mesh"))
    mesh.dofs_per_node = 2

    assert mesh.element_types == ["QUAD9", "TRI6", "TRI6"]
    assert mesh.element_material(1).E == 200.0e9

    elements = create_elements(mesh)
    total_area = 0.0
    for element in elements:
        total_area = total_area + element.area()

    assert abs(total_area - 2.0) < 1.0e-12


def test_legacy_format_still_works():
    mesh = Mesh()
    mesh.read(os.path.join(ROOT, "meshes", "triangle.mesh"))

    assert mesh.number_of_nodes() == 3
    assert mesh.element_types == ["BAR2D", "BAR2D", "BAR2D"]


def test_wrong_node_count_is_rejected():
    expect_error(VALID_MESH.replace("1 TRI 1 1   1 2 3", "1 TRI 2 1   1 2 3"))


def test_unknown_material_is_rejected():
    expect_error(VALID_MESH.replace("2 TRI 1 1   1 3 4", "2 TRI 1 7   1 3 4"))


def test_unknown_family_is_rejected():
    expect_error(VALID_MESH.replace("1 TRI 1 1", "1 HEX 1 1"))


def test_unknown_material_type_is_rejected():
    expect_error(VALID_MESH.replace("ELASTIC_PLANE_STRESS", "RUBBER"))


def test_unknown_node_is_rejected():
    expect_error(VALID_MESH.replace("1 TRI 1 1   1 2 3", "1 TRI 1 1   1 2 9"))


def test_invalid_side_is_rejected():
    # un triángulo tiene lados 1, 2 y 3
    expect_error(VALID_MESH.replace("top 1 2 2", "top 1 2 4"))


def test_unclosed_section_is_rejected():
    expect_error(VALID_MESH.replace("$EndNodes", ""))
