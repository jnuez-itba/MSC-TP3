import os

import meshio


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
