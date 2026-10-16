"""Análisis de la presa de gravedad (TP3, punto 2).

1. Cálculo manual: peso, empuje, momento estabilizante y desestabilizante
   respecto del pie, factor de seguridad al vuelco y ley trapecial de
   tensiones en la base.
2. Modelo FEM con la condición de borde elegida en la base. Comparación del
   momento de las reacciones de vínculo con el cálculo manual.
3. Figuras: tensión principal máxima sobre la deformada y tensiones en la
   base (FEM vs ley trapecial). Archivo .vtu para ParaView.
4. Con --comparar: elementos, refinamiento de malla, condiciones de borde y
   deformación plana (adoptada) vs tensión plana.

Hipótesis: estado plano de DEFORMACIÓN (ezz = 0), según indicó la cátedra.
El material se lee del archivo de malla (meshes/presa/*.mesh).

Uso:
    python analisis_presa.py                         # base empotrada, Q9
    python analisis_presa.py --support roller        # rodillo (sin fricción)
    python analisis_presa.py --support robin         # Robin: fundación elástica
    python analisis_presa.py --support robin --e-fundacion 5e9
    python analisis_presa.py --mesh meshes/presa/presa_tri6.mesh
    python analisis_presa.py --scale 5000 --field von_mises
    python analisis_presa.py --comparar --no-show    # estudio completo

El reporte se imprime y se guarda en results/presa_resultados.txt
"""

import argparse
import math
import os

import numpy

from source.mesh.mesh import Mesh
from source.mesh.generators import dam_mesh
from source.materials.linear_elastic import PLANE_STRESS
from source.dam.material import dam_material
from source.dam.geometry import DamGeometry
from source.dam.stability import stability
from source.dam.model import solve_dam
from source.dam.model import base_nodes_sorted
from source.dam.plots import plot_base_pressure
from source.systems.robin import winkler_stiffness
from source.output.postprocess import stress_fields
from source.output.postprocess import nodal_displacements
from source.output.postprocess import stress_at_point
from source.output.postprocess import principal_stresses
from source.output.paraview import write_solid_results
from source.output.plot import plot_deformed
from source.output.plot import FIELDS


parser = argparse.ArgumentParser(description="Análisis de la presa de gravedad")
parser.add_argument("--mesh", default="meshes/presa/presa_quad9.mesh")
parser.add_argument("--support", default="fixed", choices=["fixed", "roller", "robin"],
                    help="fixed = empotrada, roller = rodillo sin fricción, "
                         "robin = resortes (fundación elástica)")
parser.add_argument("--e-fundacion", type=float, default=20.0e9,
                    help="módulo de la fundación para Robin [Pa] (roca sana ~20 GPa)")
parser.add_argument("--nu-fundacion", type=float, default=0.25,
                    help="Poisson de la fundación para Robin")
parser.add_argument("--field", default="sigma_1", choices=list(FIELDS.keys()))
parser.add_argument("--scale", type=float, default=None)
parser.add_argument("--no-saturate", action="store_true",
                    help="no saturar la escala de colores en las esquinas de la base")
parser.add_argument("--comparar", action="store_true",
                    help="estudio de elementos, malla, apoyos y tensión/deformación plana")
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

SHOW = not args.no_show

SUPPORT_NAMES = {
    "fixed": "empotrada",
    "roller": "rodillo (sin fricción)",
    "robin": "Robin (fundación elástica)",
}

os.makedirs("results", exist_ok=True)
report_file = open("results/presa_resultados.txt", "w", encoding="utf-8")


def report(text=""):
    print(text)
    report_file.write(text + "\n")


def mega(value):
    return value * 1.0e-6


geometry = DamGeometry()
hand = stability(geometry)
x_toe = geometry.toe()[0]

# Rigideces de Winkler para la condición de Robin: k = E / B, G / B
K_NORMAL, K_TANGENTIAL = winkler_stiffness(args.e_fundacion, args.nu_fundacion,
                                           geometry.base)


def solve(mesh, support):
    return solve_dam(mesh, geometry.water_level(), support,
                     k_normal=K_NORMAL, k_tangential=K_TANGENTIAL)


# ======================================================================
# 1. Cálculo manual
# ======================================================================
report("=" * 72)
report("PRESA DE GRAVEDAD - " + geometry.description())
report("Hormigón: rho = {:g} kg/m3; agua: rho = {:g} kg/m3; nivel del agua = {:g} m".format(
    geometry.concrete_density, geometry.water_density, geometry.water_level()))
report("Suelo impermeable: sin subpresión. Valores por metro de presa.")
report("=" * 72)
report()
report("1) CÁLCULO MANUAL (cuerpo rígido), momentos respecto del pie (x = {:g} m)".format(x_toe))
report()
report("  Área de la sección          A  = {:10.3f} m2".format(hand["area"]))
report("  Centroide                   xc = {:10.3f} m".format(hand["centroid"][0]))
report("  Peso                        W  = {:10.4f} MN/m".format(mega(hand["weight"])))
report("  Empuje horizontal del agua  F  = {:10.4f} MN/m  (a hw/3 = {:.3f} m)".format(
    mega(hand["water_horizontal"]), geometry.water_level() / 3.0))
report()
report("  Momento estabilizante   Ms = W (x_pie - xc)  = {:10.4f} MN.m/m".format(
    mega(hand["moment_stabilizing"])))
report("  Momento desestabilizante Md = F hw / 3       = {:10.4f} MN.m/m".format(
    mega(hand["moment_overturning"])))
report("  Momento neto            Ms - Md              = {:10.4f} MN.m/m".format(
    mega(hand["moment_stabilizing"] - hand["moment_overturning"])))
report("  Factor de seguridad al vuelco  FS = Ms / Md  = {:10.3f}".format(hand["safety_factor"]))
report()
report("  Resultante sobre la base a {:.3f} m del pie (tercio central: {:g} a {:g} m) -> {}".format(
    hand["resultant_from_toe"], geometry.base / 3.0, 2.0 * geometry.base / 3.0,
    "dentro" if hand["middle_third"] else "FUERA"))
report("  Ley trapecial: compresión en el pie   = {:.4f} MPa".format(mega(hand["base_pressure_toe"])))
report("                 compresión en el talón = {:.4f} MPa".format(mega(hand["base_pressure_heel"])))
report("  Coeficiente de fricción mínimo para no deslizar  mu = F / W = {:.3f}".format(
    hand["required_friction"]))


# ======================================================================
# 2. Modelo FEM
# ======================================================================
def load_mesh(filename):
    mesh = Mesh()
    mesh.read(filename)
    return mesh


# Puntos de control: fijos en el dominio, sirven para comparar mallas.
# Están a más de 2 m de las esquinas de la base, fuera de la zona afectada
# por la singularidad del empotramiento.
FACE_POINT = (0.0, 5.0)                                   # paramento, 5 m sobre el talón
BASE_POINT = ((geometry.heel()[0] + geometry.toe()[0]) / 2.0, 0.0)   # centro de la base


def corner_distance(node):
    heel = geometry.heel()
    toe = geometry.toe()
    return min(math.hypot(node.x - heel[0], node.y - heel[1]),
               math.hypot(node.x - toe[0], node.y - toe[1]))


def fem_summary(result):
    """Valores característicos de una solución."""
    mesh = result["mesh"]
    elements = result["elements"]
    u = result["u"]

    fields = stress_fields(mesh, elements, u)
    displacements = nodal_displacements(mesh, u)

    ux_crest = max(displacements[i][0] for i in mesh.node_sets["crest"])

    s1 = fields["sigma_1"]
    i_max = int(numpy.argmax(s1))

    face = stress_at_point(elements, u, FACE_POINT[0], FACE_POINT[1])
    base = stress_at_point(elements, u, BASE_POINT[0], BASE_POINT[1])

    return {
        "fields": fields,
        "ux_crest": ux_crest,
        "s1_max": s1[i_max],
        "s1_max_at": (mesh.node(i_max).x, mesh.node(i_max).y),
        "s1_face": principal_stresses(face[0], face[1], face[2])[0],
        "syy_base": base[1],
        "dofs": mesh.number_of_dofs(),
    }


def base_pressure_curve(result, fields):
    mesh = result["mesh"]
    base = base_nodes_sorted(mesh)
    x = [mesh.node(i).x for i in base]
    q = [-fields["sigma_yy"][i] for i in base]       # compresión positiva
    return x, q


mesh = load_mesh(args.mesh)
result = solve(mesh, args.support)
summary = fem_summary(result)

report()
report("2) MODELO FEM")
report()
report("  Malla: {}  ({} elementos {}, {} grados de libertad)".format(
    args.mesh, mesh.number_of_elements(), mesh.element_types[0], summary["dofs"]))
HYPOTHESIS_NAMES = {"plane_strain": "estado plano de deformación (ezz = 0)",
                    "plane_stress": "estado plano de tensión (szz = 0)"}
material_used = mesh.element_material(0)
report("  Hormigón E = {:g} GPa, nu = {:g}; {}".format(
    material_used.E * 1.0e-9, material_used.nu, HYPOTHESIS_NAMES[material_used.hypothesis]))
report("  Base {}".format(SUPPORT_NAMES[args.support]))
if args.support == "robin":
    report("    E fundación = {:g} GPa, nu = {:g}  ->  k_n = E/B = {:.3f} GPa/m,"
           " k_t = G/B = {:.3f} GPa/m".format(args.e_fundacion * 1.0e-9, args.nu_fundacion,
                                              K_NORMAL * 1.0e-9, K_TANGENTIAL * 1.0e-9))
report()
report("  Reacciones en la base            FEM          Manual")
report("    Horizontal  Rx [MN/m]   {:12.4f}  {:12.4f}   (= -F)".format(
    mega(result["reaction_x"]), -mega(hand["water_horizontal"])))
report("    Vertical    Ry [MN/m]   {:12.4f}  {:12.4f}   (= W)".format(
    mega(result["reaction_y"]), mega(hand["normal_force"])))
report("    Momento respecto del pie [MN.m/m]")
report("      de las reacciones     {:12.4f}  {:12.4f}   (= -(Ms - Md))".format(
    mega(result["reaction_moment_toe"]),
    -mega(hand["moment_stabilizing"] - hand["moment_overturning"])))

error = abs(result["reaction_moment_toe"] + hand["moment_stabilizing"]
            - hand["moment_overturning"]) / hand["moment_stabilizing"]
report("    Diferencia relativa del momento: {:.1e}".format(error))
report()
report("  Las reacciones equilibran exactamente las cargas: su momento respecto")
report("  del pie es igual y opuesto al momento neto Ms - Md de las cargas.")
report()
report("  Desplazamiento horizontal máximo de la cresta: {:.3f} mm".format(summary["ux_crest"] * 1.0e3))
report("  sigma_1 máxima: {:.3f} MPa en ({:.2f}, {:.2f})".format(
    mega(summary["s1_max"]), summary["s1_max_at"][0], summary["s1_max_at"][1]))
report("  sigma_1 en el paramento aguas arriba, 5 m sobre el talón: {:.3f} MPa".format(
    mega(summary["s1_face"])))
report("  sigma_yy en el centro de la base: {:.3f} MPa (ley trapecial: {:.3f} MPa)".format(
    mega(summary["syy_base"]),
    -mega((hand["base_pressure_heel"] + hand["base_pressure_toe"]) / 2.0)))

szz = summary["fields"]["sigma_zz"]
s1_all = summary["fields"]["sigma_1"]
if numpy.abs(szz).max() > 0.0:
    above = int((szz > s1_all).sum())
    report("  sigma_zz = nu (sxx + syy) (fuera del plano): entre {:.3f} y {:.3f} MPa".format(
        mega(szz.min()), mega(szz.max())))
    tensile_above = int(((szz > s1_all) & (szz > 0.0)).sum())
    report("    supera a sigma_1 (en el plano) en {} de {} nodos, y en {} de ellos es".format(
        above, mesh.number_of_nodes(), tensile_above))
    report("    de tracción: solo la supera donde ambas son de compresión. La máxima")
    report("    tensión principal 3D, max(sigma_1, sigma_zz) = {:.3f} MPa, coincide con".format(
        mega(numpy.maximum(s1_all, szz).max())))
    report("    sigma_1 en el plano: el mapa de sigma_1 muestra la tracción máxima.")
report()
if args.support == "fixed":
    report("  La tracción máxima aparece en la esquina del talón. Con la base")
    report("  empotrada esa esquina es un punto singular: el valor crece al refinar")
    report("  la malla (ver --comparar). Lejos de las esquinas, el paramento aguas")
    report("  arriba está comprimido, como predice la ley trapecial (resultante")
    report("  dentro del tercio central).")
elif args.support == "roller":
    report("  Rodillo sin fricción: el único nodo con u = 0 (talón) toma todo el")
    report("  empuje del agua como carga puntual, de ahí el pico de tensión en el")
    report("  talón. Es un caso límite (fricción nula), no el comportamiento real.")
else:
    report("  Robin: la fundación se deforma con la presa. No hay esquina rígida,")
    report("  así que desaparece la singularidad del talón; la tensión máxima es")
    report("  chica y la presión en la base se parece a la ley trapecial. La cresta")
    report("  se desplaza más porque se suma la rotación sobre la fundación.")


# ======================================================================
# 3. Figuras
# ======================================================================
tag = args.support + "_" + mesh.element_types[0].lower()

write_solid_results("results/presa_" + tag + ".vtu", mesh, result["elements"], result["u"])

# Escala saturada: el límite de colores se toma con los nodos a más de 1 m
# de las esquinas de la base, donde están los picos singulares.
color_limit = None
if not args.no_saturate and args.field in summary["fields"]:
    values = summary["fields"][args.field]
    far_values = [abs(values[node.global_id]) for node in mesh.nodes
                  if corner_distance(node) > 1.0]
    color_limit = mega(max(far_values))

scale = plot_deformed(
    mesh, result["elements"], result["u"],
    field=args.field,
    scale=args.scale,
    filename="results/presa_" + args.field + "_" + tag + ".png",
    show=SHOW,
    title="Presa - base " + SUPPORT_NAMES[args.support],
    color_limit=color_limit,
)

x, q = base_pressure_curve(result, summary["fields"])
plot_base_pressure(
    geometry, hand,
    [("FEM, base " + SUPPORT_NAMES[args.support], x, q)],
    filename="results/presa_base_" + tag + ".png",
    show=SHOW,
    clip=2.0,
)

report()
report("  Figuras: results/presa_{}_{}.png, results/presa_base_{}.png".format(args.field, tag, tag))
report("  ParaView: results/presa_{}.vtu   (factor de escala usado: {:g})".format(tag, scale))


# ======================================================================
# 4. Estudio comparativo
# ======================================================================
if args.comparar:
    material = dam_material()

    report()
    report("=" * 72)
    report("3) ESTUDIO COMPARATIVO")
    report("=" * 72)

    # --- 4 tipos de elemento ---------------------------------------
    report()
    report("  a) Tipo de elemento (malla de ~1 m, base empotrada)")
    report()
    report("    {:<7}{:>7}{:>12}{:>13}{:>15}{:>14}".format(
        "Elem", "GdL", "ux cresta", "s1 máx", "s1 paramento", "syy centro"))
    report("    {:<7}{:>7}{:>12}{:>13}{:>15}{:>14}".format(
        "", "", "[mm]", "talón [MPa]", "y=5 m [MPa]", "base [MPa]"))

    for element_type in ["TRI3", "QUAD4", "TRI6", "QUAD9"]:
        m = load_mesh("meshes/presa/presa_" + element_type.lower() + ".mesh")
        s = fem_summary(solve_dam(m, geometry.water_level(), "fixed"))
        report("    {:<7}{:>7}{:>12.4f}{:>13.3f}{:>15.3f}{:>14.3f}".format(
            element_type, s["dofs"], s["ux_crest"] * 1.0e3, mega(s["s1_max"]),
            mega(s["s1_face"]), mega(s["syy_base"])))

    # --- refinamiento de malla -------------------------------------
    report()
    report("  b) Refinamiento de malla (Q9, base empotrada)")
    report()
    report("    {:<9}{:>7}{:>12}{:>13}{:>15}{:>14}".format(
        "Tamaño", "GdL", "ux cresta", "s1 máx", "s1 paramento", "syy centro"))

    for size in [3.0, 2.0, 1.0, 0.75]:
        m = dam_mesh(material, "QUAD9", size, geometry.height, geometry.base,
                     geometry.crest, geometry.freeboard)
        s = fem_summary(solve_dam(m, geometry.water_level(), "fixed"))
        report("    {:<9}{:>7}{:>12.4f}{:>13.3f}{:>15.3f}{:>14.3f}".format(
            format(size, "g") + " m", s["dofs"], s["ux_crest"] * 1.0e3,
            mega(s["s1_max"]), mega(s["s1_face"]), mega(s["syy_base"])))

    report()
    report("    El desplazamiento y las tensiones en puntos fijos lejos de las")
    report("    esquinas convergen.")
    report("    sigma_1 máxima (en el talón) crece al refinar: es una singularidad")
    report("    del empotramiento en la esquina, no un valor físico.")

    # --- condiciones de borde --------------------------------------
    report()
    report("  c) Condición de borde en la base (malla Q9 de ~1 m)")
    report()
    report("    {:<28}{:>9}{:>9}{:>10}{:>11}{:>9}{:>11}{:>11}".format(
        "Base", "Rx [MN]", "Ry [MN]", "M pie", "ux cresta", "s1 máx", "s1 param.", "syy centro"))

    curves = []
    for support in ["fixed", "roller", "robin"]:
        m = load_mesh(args.mesh)
        r = solve(m, support)
        s = fem_summary(r)
        report("    {:<28}{:>9.4f}{:>9.4f}{:>10.3f}{:>11.4f}{:>9.3f}{:>11.3f}{:>11.3f}".format(
            SUPPORT_NAMES[support], mega(r["reaction_x"]), mega(r["reaction_y"]),
            mega(r["reaction_moment_toe"]), s["ux_crest"] * 1.0e3, mega(s["s1_max"]),
            mega(s["s1_face"]), mega(s["syy_base"])))
        x, q = base_pressure_curve(r, s["fields"])
        curves.append(("FEM, base " + SUPPORT_NAMES[support], x, q))

        if support == "roller":
            heel = m.node_sets["heel"][0]
            roller_heel_reaction = r["reactions"][m.node_global_dofs(heel)[0]]

    report()
    report("    Empotrada = fricción infinita; rodillo = fricción nula: son los dos")
    report("    casos límite de un modelo lineal (la fricción de Coulomb es no lineal).")
    report("    Rodillo: el nodo del talón toma Rx = {:.4f} MN, TODO el empuje del agua".format(
        mega(roller_heel_reaction)))
    report("    en un punto. Sin fricción no hay otra forma de equilibrar el empuje.")
    report("    Como la fricción necesaria (mu = {:.2f}) es menor que la disponible para".format(
        hand["required_friction"]))
    report("    hormigón sobre roca (~0.6-0.8), la presa no desliza y el caso")
    report("    empotrado es el más representativo.")
    report("    Robin (E fundación = {:g} GPa): la fundación se deforma, la cresta se".format(
        args.e_fundacion * 1.0e-9))
    report("    mueve más, desaparece la singularidad del talón y la presión en la")
    report("    base se acerca a la ley trapecial.")

    plot_base_pressure(geometry, hand, curves,
                       filename="results/presa_base_comparacion.png",
                       show=SHOW, clip=2.0)

    # --- deformación plana (adoptada) vs tensión plana ---------------
    report()
    report("  d) Deformación plana (adoptada) vs tensión plana (Q9 ~1 m, base empotrada)")
    report()
    for name, mat in [("deformación plana", material),
                      ("tensión plana", dam_material(PLANE_STRESS))]:
        m = dam_mesh(mat, "QUAD9", 1.0, geometry.height, geometry.base,
                     geometry.crest, geometry.freeboard)
        s = fem_summary(solve_dam(m, geometry.water_level(), "fixed"))
        report("    {:<20} ux cresta = {:.4f} mm   s1 paramento (y=5 m) = {:.3f} MPa".format(
            name, s["ux_crest"] * 1.0e3, mega(s["s1_face"])))
    report()
    report("    La presa es muy larga en z: cada sección no puede deformarse fuera del")
    report("    plano (ezz = 0). Deformación plana es más rígida: E' = E/(1-nu^2).")

    report()
    report("  Figura: results/presa_base_comparacion.png")

report_file.close()
