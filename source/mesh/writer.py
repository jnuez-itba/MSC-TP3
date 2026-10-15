"""Escritura de mallas en el formato por secciones (ver README.md)."""

import os

from source.materials.factory import material_type_name


def format_number(value):
    return format(value, ".15g")


def write_ids(file, ids, per_line=12):
    for k in range(0, len(ids), per_line):
        file.write(" ".join(str(i) for i in ids[k:k + per_line]) + "\n")


def write_mesh(filename, mesh, comment=None):
    """Escribe la malla (nodos, elementos, materiales, espesor y conjuntos)."""
    directory = os.path.dirname(filename)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

    file = open(filename, "w")

    if comment is not None:
        for line in comment.splitlines():
            file.write("# " + line + "\n")
        file.write("\n")

    file.write("$Parameters\n")
    file.write("thickness " + format_number(mesh.thickness) + "\n")
    file.write("$EndParameters\n\n")

    file.write("$Materials\n")
    file.write("# id  tipo  E  nu  densidad\n")
    for material_id in sorted(mesh.materials):
        material = mesh.materials[material_id]
        file.write(" ".join([
            str(material_id),
            material_type_name(material),
            format_number(material.E),
            format_number(material.nu),
            format_number(material.density),
        ]) + "\n")
    file.write("$EndMaterials\n\n")

    file.write("$Nodes\n")
    file.write(str(mesh.number_of_nodes()) + "\n")
    file.write("# id  x  y\n")
    for node in mesh.nodes:
        file.write(str(node.global_id + 1) + " " + format_number(node.x)
                   + " " + format_number(node.y) + "\n")
    file.write("$EndNodes\n\n")

    file.write("$Elements\n")
    file.write(str(mesh.number_of_elements()) + "\n")
    file.write("# id  familia  orden  material  nodos\n")
    for e in range(mesh.number_of_elements()):
        nodes = " ".join(str(i + 1) for i in mesh.connectivity[e])
        file.write(str(e + 1) + " " + mesh.element_families[e] + " "
                   + str(mesh.element_orders[e]) + " "
                   + str(mesh.element_material_ids[e]) + " " + nodes + "\n")
    file.write("$EndElements\n")

    if len(mesh.node_sets) > 0:
        file.write("\n$NodeSets\n")
        file.write("# nombre  cantidad  nodos\n")
        for name in mesh.node_sets:
            ids = [i + 1 for i in mesh.node_sets[name]]
            file.write(name + " " + str(len(ids)) + "\n")
            write_ids(file, ids)
        file.write("$EndNodeSets\n")

    if len(mesh.side_sets) > 0:
        file.write("\n$SideSets\n")
        file.write("# nombre  cantidad  (elemento lado) ...\n")
        for name in mesh.side_sets:
            sides = mesh.side_sets[name]
            file.write(name + " " + str(len(sides)) + "\n")
            for element_id, side in sides:
                file.write(str(element_id + 1) + " " + str(side + 1) + "\n")
        file.write("$EndSideSets\n")

    file.close()
