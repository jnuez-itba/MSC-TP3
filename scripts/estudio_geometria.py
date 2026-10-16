"""Estudio de variaciones de geometría para aumentar el factor de seguridad.

Recorre cuatro familias de variantes (ver source/dam/study.py):
    base     base más ancha hacia aguas abajo
    talon    talón extendido aguas arriba (talud, base 18 + a)
    cresta   cresta más ancha
    talud    talud aguas arriba con la misma base (sin agregar hormigón)

Para cada una calcula la estabilidad a mano (FS al vuelco, tercio central,
presiones en la base, fricción necesaria) y la verifica con FEM (Q9, base
empotrada): momento de las reacciones, desplazamiento de la cresta y
tensión principal en el paramento aguas arriba.

Uso:
    python scripts/estudio_geometria.py
    python scripts/estudio_geometria.py --sin-fem          # solo cálculo manual (rápido)
    python scripts/estudio_geometria.py --no-show

El reporte queda en results/estudio_geometria.txt
"""

# Los scripts están en scripts/: se ejecutan desde la raíz del repositorio,
# para que encuentren el paquete source/ y las rutas meshes/ y results/.
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import argparse
import math
import os

from source.dam.study import FAMILIES
from source.dam.study import SWEEPS
from source.dam.study import parameter_label
from source.dam.study import run_study
from source.dam.study import best_at_area
from source.dam.study import evaluate
from source.dam.geometry import DamGeometry
from source.dam.plots import plot_safety_factor_vs_area
from source.dam.plots import plot_sections
from source.output.plot import plot_deformed


parser = argparse.ArgumentParser(description="Estudio de geometrías de la presa")
parser.add_argument("--sin-fem", action="store_true", help="solo cálculo manual")
parser.add_argument("--size", type=float, default=1.5, help="tamaño de elemento FEM [m]")
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

SHOW = not args.no_show
FEM = not args.sin_fem

os.makedirs("results", exist_ok=True)
report_file = open("results/estudio_geometria.txt", "w", encoding="utf-8")


def report(text=""):
    print(text)
    report_file.write(text + "\n")


def mega(value):
    return value * 1.0e-6


original = evaluate(DamGeometry(), fem=FEM, element_size=args.size)
results = run_study(fem=FEM, element_size=args.size)

report("=" * 78)
report("ESTUDIO DE VARIACIONES DE GEOMETRÍA")
report("Presa del enunciado: A = {:.0f} m2, FS al vuelco = {:.2f}".format(
    original["area"], original["safety_factor"]))
report("=" * 78)
report()
report("Columnas:")
report("  A       área de hormigón (volumen por metro de presa)")
report("  FS      factor de seguridad al vuelco = Ms / Md (respecto del pie)")
report("  xR      distancia de la resultante al pie; 'tc' = dentro del tercio central")
report("  q tal.  presión en el talón, ley trapecial (negativa = tracción)")
report("  q pie   presión en el pie, ley trapecial")
report("  mu      fricción mínima para no deslizar")
if FEM:
    report("  err M   diferencia entre el momento de las reacciones FEM y Ms - Md")
    report("  ux      desplazamiento horizontal de la cresta (FEM)")
    report("  s1 par. tensión principal máxima en el paramento aguas arriba a 5 m (FEM)")

for family in SWEEPS:
    report()
    report(FAMILIES[family])
    header = "  {:<20}{:>7}{:>7}{:>9}{:>4}{:>9}{:>8}{:>7}".format(
        "parámetro", "A", "FS", "xR [m]", "", "q tal.", "q pie", "mu")
    if FEM:
        header = header + "{:>10}{:>9}{:>9}".format("err M", "ux [mm]", "s1 par.")
    report(header)

    for value, r in results[family]:
        line = "  {:<20}{:>7.0f}{:>7.2f}{:>9.2f}{:>4}{:>9.3f}{:>8.3f}{:>7.3f}".format(
            parameter_label(family, value), r["area"], r["safety_factor"],
            r["resultant_from_toe"], "tc" if r["middle_third"] else "NO",
            mega(r["q_heel"]), mega(r["q_toe"]), r["required_friction"])
        if FEM:
            line = line + "{:>10.1e}{:>9.3f}{:>9.3f}".format(
                r["fem_moment_error"], r["ux_crest"] * 1.0e3, mega(r["s1_face"]))
        report(line)


# ----------------------------------------------------------------------
# Comparación con la misma cantidad de hormigón
# ----------------------------------------------------------------------
target_area = 375.0
same_area = best_at_area(results, target_area)

report()
report("=" * 78)
report("COMPARACIÓN CON LA MISMA ÁREA DE HORMIGÓN: A = {:.0f} m2 (+{:.0f} %)".format(
    target_area, 100.0 * (target_area / original["area"] - 1.0)))
report("=" * 78)
report()
report("  {:<50}{:>7}{:>9}{:>9}{:>8}".format("variante", "FS", "q tal.", "q pie", "mu"))
for family, value, r in same_area:
    report("  {:<50}{:>7.2f}{:>9.3f}{:>9.3f}{:>8.3f}".format(
        FAMILIES[family] + ", " + parameter_label(family, value).split(" (")[0],
        r["safety_factor"], mega(r["q_heel"]), mega(r["q_toe"]), r["required_friction"]))

report()
report("Conclusiones:")
report("  - Alargar la base es lo que más aumenta el FS al vuelco por m2 de")
report("    hormigón. Al correr el pie (el punto de vuelco) hacia aguas abajo crece")
report("    el brazo de TODO el peso de la presa, no solo el del hormigón agregado.")
report("    Además deja la presión en la base más pareja entre talón y pie.")
report("  - Extender el talón aguas arriba da casi el mismo FS: el pie no se mueve,")
report("    pero el hormigón nuevo queda lejos de él y el agua que queda sobre el")
report("    talud suma peso. Por ese peso extra es la variante que menos fricción")
report("    necesita para no deslizar.")
report("  - Ensanchar la cresta suma peso sin mover el pie: mejora poco.")
report("  - Inclinar el paramento sin agregar hormigón NO mejora: con la base fija el")
report("    pie se corre aguas arriba y el FS baja; con talud grande la resultante")
report("    sale del tercio central y el talón queda traccionado.")
report("  - En todas las variantes el momento de las reacciones del FEM coincide con")
report("    Ms - Md del cálculo manual (err M ~ 1e-13).")


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
plot_safety_factor_vs_area(results, FAMILIES,
                           filename="results/estudio_fs_vs_area.png",
                           show=SHOW, original=original)

designs = [("Presa del enunciado", original["geometry"], original["safety_factor"])]
for family, value, r in same_area:
    designs.append((FAMILIES[family] + ", " + parameter_label(family, value).split(" (")[0],
                    r["geometry"], r["safety_factor"]))

plot_sections(designs, filename="results/estudio_secciones.png", show=SHOW)

if FEM:
    # tensión principal sobre la deformada de la mejor variante a igual área
    family, value, r = same_area[0]
    mesh = r["mesh"]
    solution = r["solution"]

    heel = r["geometry"].heel()
    toe = r["geometry"].toe()
    from source.output.postprocess import stress_fields
    s1 = stress_fields(mesh, solution["elements"], solution["u"])["sigma_1"]
    far = [abs(s1[n.global_id]) for n in mesh.nodes
           if math.hypot(n.x - heel[0], n.y - heel[1]) > 1.0
           and math.hypot(n.x - toe[0], n.y - toe[1]) > 1.0]

    plot_deformed(mesh, solution["elements"], solution["u"], field="sigma_1",
                  filename="results/estudio_sigma_1_" + family + ".png", show=SHOW,
                  title=FAMILIES[family] + ", " + parameter_label(family, value),
                  color_limit=mega(max(far)))

report()
report("Figuras: results/estudio_fs_vs_area.png, results/estudio_secciones.png")
if FEM:
    report("         results/estudio_sigma_1_" + same_area[0][0] + ".png")

report_file.close()
