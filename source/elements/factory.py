from source.elements.tri3 import Tri3
from source.elements.quad4 import Quad4
from source.elements.tri6 import Tri6
from source.elements.quad9 import Quad9


# Tipo de elemento -> clase
# (Mesh traduce familia + orden del archivo a estos nombres)
ELEMENT_TYPES = {
    "TRI3": Tri3,
    "QUAD4": Quad4,
    "TRI6": Tri6,
    "QUAD9": Quad9,
}


def create_element(element_type, global_id, node_global_ids, mesh, material, thickness):
    """Crea el elemento que corresponde al tipo indicado."""
    if element_type not in ELEMENT_TYPES:
        raise ValueError("Unknown element type: " + element_type)

    element_class = ELEMENT_TYPES[element_type]

    return element_class(global_id, node_global_ids, mesh, material, thickness)


def create_elements(mesh):
    """Crea todos los elementos de la malla, cada uno con el material y el
    espesor que indica el archivo."""
    elements = []

    for e in range(mesh.number_of_elements()):
        element = create_element(
            mesh.element_types[e],
            mesh.element_global_ids[e],
            mesh.connectivity[e],
            mesh,
            mesh.element_material(e),
            mesh.thickness,
        )

        elements.append(element)

    return elements
