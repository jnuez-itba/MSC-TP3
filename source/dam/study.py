"""Estudio de variaciones de geometría de la presa.

Familias de variantes (el resto de la geometría queda como en el enunciado):

  "base"      ensanchar la base hacia aguas abajo (el pie se aleja)
  "talon"     extender el talón aguas arriba con un talud: la base pasa a
              ser 18 + a y el pie queda en el mismo lugar
  "cresta"    ensanchar la cresta
  "talud"     inclinar el paramento aguas arriba SIN agregar hormigón (misma
              base de 18 m): solo redistribuye la masa

Para cada variante se calcula la estabilidad a mano y, opcionalmente, con
el modelo FEM (base empotrada).
"""

import math

from source.dam.geometry import DamGeometry
from source.dam.stability import stability
from source.dam.model import solve_dam
from source.mesh.generators import dam_mesh
from source.materials.linear_elastic import LinearElastic
from source.output.postprocess import nodal_displacements
from source.output.postprocess import stress_at_point
from source.output.postprocess import principal_stresses


FAMILIES = {
    "base": "Base más ancha (pie aguas abajo)",
    "talon": "Talón extendido aguas arriba",
    "cresta": "Cresta más ancha",
    "talud": "Talud aguas arriba, misma base",
}


def make_variant(family, value):
    """Geometría de una variante. 'value' es el parámetro de la familia [m]."""
    if family == "base":
        return DamGeometry(base=value)

    if family == "talon":
        return DamGeometry(base=18.0 + value, upstream_batter=value)

    if family == "cresta":
        return DamGeometry(crest=value)

    if family == "talud":
        return DamGeometry(upstream_batter=value)

    raise ValueError("Unknown family: " + family)


def parameter_label(family, value):
    if family == "base":
        return "B = {:g} m".format(value)
    if family == "talon":
        return "a = {:g} m (B = {:g} m)".format(value, 18.0 + value)
    if family == "cresta":
        return "c = {:g} m".format(value)
    return "a = {:g} m".format(value)


# Valores de cada familia que se recorren en el estudio
SWEEPS = {
    "base": [16.0, 18.0, 20.0, 22.0, 24.0],
    "talon": [0.0, 2.0, 4.0, 6.0],
    "cresta": [3.0, 5.0, 7.0, 9.0],
    "talud": [0.0, 2.0, 4.0, 6.0],
}


def face_x(geometry, y):
    """x del paramento aguas arriba a la altura y."""
    a = geometry.upstream_batter
    return -a + a * y / geometry.height


def evaluate(geometry, fem=True, element_type="QUAD9", element_size=1.5,
             E=21.0e9, nu=0.2):
    """Estabilidad a mano y, si fem=True, resultados del modelo FEM."""
    hand = stability(geometry)

    result = {
        "geometry": geometry,
        "area": hand["area"],
        "safety_factor": hand["safety_factor"],
        "resultant_from_toe": hand["resultant_from_toe"],
        "middle_third": hand["middle_third"],
        "q_heel": hand["base_pressure_heel"],
        "q_toe": hand["base_pressure_toe"],
        "required_friction": hand["required_friction"],
        "hand": hand,
    }

    if not fem:
        return result

    material = LinearElastic(E, nu, geometry.concrete_density)
    mesh = dam_mesh(material, element_type, element_size, geometry.height,
                    geometry.base, geometry.crest, geometry.freeboard,
                    geometry.upstream_batter)

    solution = solve_dam(mesh, geometry.water_level(), "fixed")

    net = hand["moment_stabilizing"] - hand["moment_overturning"]
    moment_error = abs(solution["reaction_moment_toe"] + net) / net

    displacements = nodal_displacements(mesh, solution["u"])
    ux_crest = max(displacements[i][0] for i in mesh.node_sets["crest"])

    # Tensión principal máxima en el paramento aguas arriba, 5 m sobre la base
    y = 5.0
    s = stress_at_point(solution["elements"], solution["u"], face_x(geometry, y), y)
    s1_face = principal_stresses(s[0], s[1], s[2])[0]

    result.update({
        "fem_moment_error": moment_error,
        "ux_crest": ux_crest,
        "s1_face": s1_face,
        "mesh": mesh,
        "solution": solution,
    })

    return result


def run_study(fem=True, element_type="QUAD9", element_size=1.5):
    """Recorre todas las familias. Devuelve {familia: [(valor, resultado)]}."""
    results = {}

    for family in SWEEPS:
        results[family] = []
        for value in SWEEPS[family]:
            geometry = make_variant(family, value)
            results[family].append(
                (value, evaluate(geometry, fem, element_type, element_size)))

    return results


def best_at_area(results, area, tolerance=1.0e-6):
    """Variantes de cada familia con un área dada (para comparar con la misma
    cantidad de hormigón)."""
    chosen = []

    for family in results:
        for value, result in results[family]:
            if math.isclose(result["area"], area, rel_tol=tolerance):
                chosen.append((family, value, result))

    chosen.sort(key=lambda item: -item[2]["safety_factor"])

    return chosen
