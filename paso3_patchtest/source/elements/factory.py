from source.elements.tri3 import Tri3
from source.elements.quad4 import Quad4


# Tipo de elemento en el archivo de malla -> clase
ELEMENT_TYPES = {
    "TRI3": Tri3,
    "QUAD4": Quad4,
}


def create_element(element_type, global_id, node_global_ids, mesh, material, thickness):
    """Crea el elemento que corresponde al texto del archivo de malla."""
    if element_type not in ELEMENT_TYPES:
        raise ValueError("Unknown element type: " + element_type)

    element_class = ELEMENT_TYPES[element_type]

    return element_class(global_id, node_global_ids, mesh, material, thickness)


def create_elements(mesh, material, thickness):
    """Crea todos los elementos de la malla con un mismo material y espesor."""
    elements = []

    for e in range(mesh.number_of_elements()):
        element = create_element(
            mesh.element_types[e],
            mesh.element_global_ids[e],
            mesh.connectivity[e],
            mesh,
            material,
            thickness,
        )

        elements.append(element)

    return elements
