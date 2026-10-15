import os

import meshio
import numpy

from source.output.postprocess import nodal_displacements
from source.output.postprocess import stress_fields


# Tipo de elemento -> nombre de celda de meshio / VTK.
# El orden local de nodos de Tri6 y Q9 es el mismo que usa VTK
# (vértices, lados, centro), así que la conectividad se escribe tal cual.
VTK_CELL_TYPES = {
    "TRI3": "triangle",
    "QUAD4": "quad",
    "TRI6": "triangle6",
    "QUAD9": "quad9",
}


def line_cells(mesh):
    lines = []

    for connectivity in mesh.connectivity:
        lines.append([connectivity[0], connectivity[1]])

    return [("line", lines)]


def write(filename, points, cells, displacement, stress):
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data={"displacement": displacement},
        cell_data={"stress": [stress]},
    )

    output_mesh.write(filename)


def write_results(filename, mesh, elements, u):
    """Resultados de barras (código de la cátedra)."""
    points = []
    displacement = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)

        points.append([node.x, node.y, 0.0])
        displacement.append([u[dofs[0]], u[dofs[1]], 0.0])

    stress = []

    for element in elements:
        stress.append(element.axial_stress(u))

    write(filename, points, line_cells(mesh), displacement, stress)


def write_solid_results(filename, mesh, elements, u):
    """Resultados de sólidos 2D para ParaView (.vtu).

    Datos nodales:
        displacement                      [u, v, 0]  (para "Warp By Vector")
        sigma_xx, sigma_yy, tau_xy        tensiones promediadas en los nodos
        sigma_1, sigma_2                  tensiones principales
        von_mises                         tensión equivalente
    Datos por celda:
        material                          id del material de cada elemento

    Para ver la deformada en ParaView: filtro "Warp By Vector" con el vector
    "displacement" y el factor de escala que se quiera.
    """
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    points = numpy.array([[node.x, node.y, 0.0] for node in mesh.nodes])

    displacements = nodal_displacements(mesh, u)
    displacement_3d = numpy.zeros((mesh.number_of_nodes(), 3))
    displacement_3d[:, 0:2] = displacements

    point_data = {"displacement": displacement_3d}
    point_data.update(stress_fields(mesh, elements, u))

    # meshio agrupa las celdas en bloques de un mismo tipo
    blocks = {}
    materials = {}

    for e in range(len(elements)):
        cell_type = VTK_CELL_TYPES[mesh.element_types[e]]

        if cell_type not in blocks:
            blocks[cell_type] = []
            materials[cell_type] = []

        blocks[cell_type].append(mesh.connectivity[e])

        if len(mesh.element_material_ids) > 0:
            materials[cell_type].append(mesh.element_material_ids[e])
        else:
            materials[cell_type].append(0)

    cells = []
    material_data = []

    for cell_type in blocks:
        cells.append((cell_type, numpy.array(blocks[cell_type])))
        material_data.append(numpy.array(materials[cell_type]))

    output_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data={"material": material_data},
    )

    output_mesh.write(filename)
