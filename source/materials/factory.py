from source.materials.linear_elastic import LinearElastic
from source.materials.linear_elastic import PLANE_STRESS
from source.materials.linear_elastic import PLANE_STRAIN


# Tipo de material en el archivo de malla -> (clase, hipótesis)
# Parámetros que se leen del archivo, en orden: E nu densidad
MATERIAL_TYPES = {
    "ELASTIC_PLANE_STRESS": PLANE_STRESS,
    "ELASTIC_PLANE_STRAIN": PLANE_STRAIN,
}


def create_material(material_type, parameters):
    """Crea un material a partir del tipo y la lista de parámetros del archivo."""
    if material_type not in MATERIAL_TYPES:
        raise ValueError("Unknown material type: " + material_type)

    if len(parameters) != 3:
        raise ValueError(
            "Material " + material_type + " expects 3 parameters: E nu density"
        )

    E = float(parameters[0])
    nu = float(parameters[1])
    density = float(parameters[2])

    return LinearElastic(E, nu, density, MATERIAL_TYPES[material_type])
