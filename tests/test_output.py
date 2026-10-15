import os
import tempfile

import numpy
import matplotlib

matplotlib.use("Agg")   # sin ventanas durante los tests

import meshio

from source.mesh.mesh import Mesh
from source.elements.factory import create_elements
from source.materials.linear_elastic import LinearElastic
from source.output.postprocess import principal_stresses
from source.output.postprocess import von_mises
from source.output.postprocess import nodal_stresses
from source.output.paraview import write_solid_results
from source.output.paraview import VTK_CELL_TYPES
from source.output.plot import plot_deformed
from source.output.plot import nice_number
from source.output.plot import SUBTRIANGLES
from source.verification.patch import exact_displacement
from source.verification.patch import exact_strain

from tests.helpers import rectangle_mesh
from tests.test_elements import CASES, make_element, polygon_area, VERTEX_COUNT


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATCH_MESHES = ["tri3", "quad4", "tri6", "quad9"]


def test_principal_stresses_pure_shear():
    # Corte puro tau: s1 = tau, s2 = -tau, a 45 grados
    s1, s2, theta = principal_stresses(0.0, 0.0, 5.0)

    assert abs(s1 - 5.0) < 1.0e-12
    assert abs(s2 + 5.0) < 1.0e-12
    assert abs(theta - numpy.pi / 4.0) < 1.0e-12


def test_principal_stresses_invariants():
    sxx, syy, txy = 30.0, -10.0, 12.0
    s1, s2, theta = principal_stresses(sxx, syy, txy)

    # la traza y el determinante no dependen de la orientación
    assert abs((s1 + s2) - (sxx + syy)) < 1.0e-12
    assert abs(s1 * s2 - (sxx * syy - txy * txy)) < 1.0e-9
    assert s1 >= s2

    # rotando el tensor un ángulo theta tiene que quedar diagonal
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    rotated_shear = (syy - sxx) * s * c + txy * (c * c - s * s)
    assert abs(rotated_shear) < 1.0e-12


def test_von_mises():
    assert abs(von_mises(100.0, 0.0, 0.0) - 100.0) < 1.0e-12             # uniaxial
    assert abs(von_mises(0.0, 0.0, 10.0) - 10.0 * numpy.sqrt(3.0)) < 1.0e-12  # corte


def test_nodal_stresses_exact_for_linear_field():
    # Con un campo lineal, la tensión es constante: todos los nodos tienen
    # que dar exactamente D * epsilon, para los 4 elementos.
    for name in PATCH_MESHES:
        mesh = Mesh()
        mesh.read(os.path.join(ROOT, "meshes", "patch", "patch_" + name + ".mesh"))
        mesh.dofs_per_node = 2
        elements = create_elements(mesh)

        u = numpy.zeros(mesh.number_of_dofs())
        for node in mesh.nodes:
            dofs = mesh.node_global_dofs(node.global_id)
            u[dofs[0]], u[dofs[1]] = exact_displacement(node.x, node.y)

        expected = elements[0].material.D @ exact_strain()
        stresses = nodal_stresses(mesh, elements, u)

        for row in stresses:
            assert numpy.allclose(row, expected, rtol=1.0e-10)


def test_subtriangles_cover_each_element():
    # Los triángulos que se usan para dibujar tienen que cubrir exactamente
    # el elemento (suma de áreas = área del elemento).
    names = {"Tri3": "TRI3", "Quad4": "QUAD4", "Tri6": "TRI6", "Quad9": "QUAD9"}

    for element_class, coordinates in CASES:
        element_type = names[element_class.__name__]
        vertices = coordinates[:VERTEX_COUNT[element_class]]

        total = 0.0
        for a, b, c in SUBTRIANGLES[element_type]:
            total = total + polygon_area([coordinates[a], coordinates[b], coordinates[c]])

        assert abs(total - polygon_area(vertices)) < 1.0e-12


def test_paraview_file_roundtrip():
    material = LinearElastic(21.0e9, 0.2, 2400.0)

    for element_type in ["TRI3", "QUAD4", "TRI6", "QUAD9"]:
        mesh = rectangle_mesh(2.0, 1.0, 2, 2, element_type, material)
        elements = create_elements(mesh)

        u = numpy.arange(mesh.number_of_dofs()) * 1.0e-6

        handle, filename = tempfile.mkstemp(suffix=".vtu")
        os.close(handle)

        try:
            write_solid_results(filename, mesh, elements, u)
            result = meshio.read(filename)
        finally:
            os.remove(filename)

        assert len(result.points) == mesh.number_of_nodes()
        assert result.cells[0].type == VTK_CELL_TYPES[element_type]
        assert len(result.cells[0].data) == mesh.number_of_elements()

        for name in ["displacement", "sigma_1", "sigma_2", "von_mises",
                     "sigma_xx", "sigma_yy", "tau_xy"]:
            assert name in result.point_data

        displacement = result.point_data["displacement"]
        assert numpy.allclose(displacement[:, 0], u[0::2])
        assert numpy.allclose(displacement[:, 1], u[1::2])


def test_nice_number():
    assert nice_number(1.2) == 1.0
    assert nice_number(2.7) == 2.0
    assert nice_number(430.0) == 500.0
    assert nice_number(0.08) == 0.1


def test_plot_deformed_saves_figure():
    material = LinearElastic(21.0e9, 0.2, 2400.0)
    mesh = rectangle_mesh(2.0, 1.0, 2, 2, "QUAD9", material)
    elements = create_elements(mesh)
    u = numpy.arange(mesh.number_of_dofs()) * 1.0e-6

    handle, filename = tempfile.mkstemp(suffix=".png")
    os.close(handle)

    try:
        scale = plot_deformed(mesh, elements, u, field="von_mises", scale=10.0,
                              filename=filename, show=False)
        assert scale == 10.0
        assert os.path.getsize(filename) > 0

        # escala automática
        scale = plot_deformed(mesh, elements, u, field="displacement",
                              filename=filename, show=False)
        assert scale > 0.0
    finally:
        os.remove(filename)
