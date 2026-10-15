"""Genera la malla de la presa de gravedad y la guarda en el formato del TP.

Uso:
    python generar_malla_presa.py                          # Q9, elementos de 1 m
    python generar_malla_presa.py --element TRI6 --size 0.5
    python generar_malla_presa.py --batter 3               # talud aguas arriba de 3 m
    python generar_malla_presa.py --no-show                # sin abrir la figura

Por defecto escribe meshes/presa/presa_<elemento>.mesh y la figura
results/malla_presa_<elemento>.png.
"""

import argparse

from source.mesh.generators import dam_mesh
from source.mesh.writer import write_mesh
from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.factory import create_elements
from source.output.plot import plot_mesh


# ----------------------------------------------------------------------
# Datos del enunciado
# ----------------------------------------------------------------------
HEIGHT = 30.0       # altura [m]
BASE = 18.0         # ancho de la base [m]
FREEBOARD = 2.0     # borde libre [m]  -> nivel del agua = 28 m
CREST = 3.0         # ancho de la cresta [m]

E = 21.0e9          # hormigón H21 [Pa]
NU = 0.2
DENSITY = 2400.0    # [kg/m3]


parser = argparse.ArgumentParser(description="Malla de la presa de gravedad")
parser.add_argument("--element", default="QUAD9",
                    choices=["TRI3", "QUAD4", "TRI6", "QUAD9"])
parser.add_argument("--size", type=float, default=1.0,
                    help="tamaño aproximado de elemento [m]")
parser.add_argument("--batter", type=float, default=0.0,
                    help="talud aguas arriba: corrimiento del talón [m]")
parser.add_argument("--base", type=float, default=BASE, help="ancho de la base [m]")
parser.add_argument("--crest", type=float, default=CREST, help="ancho de la cresta [m]")
parser.add_argument("--out", default=None, help="archivo de salida")
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

if args.out is None:
    args.out = "meshes/presa/presa_" + args.element.lower() + ".mesh"


material = LinearElastic(E, NU, DENSITY)

mesh = dam_mesh(
    material,
    element_type=args.element,
    element_size=args.size,
    height=HEIGHT,
    base=args.base,
    crest=args.crest,
    freeboard=FREEBOARD,
    upstream_batter=args.batter,
)

comment = (
    "Presa de gravedad - malla generada con generar_malla_presa.py\n"
    "H = {} m, base = {} m, cresta = {} m, borde libre = {} m, talud aguas arriba = {} m\n"
    "Elemento {}, tamaño ~{} m\n"
    "NodeSets: base, heel (talón), toe (pie), upstream, downstream, crest\n"
    "SideSets: base, upstream, upstream_wet (bajo el agua), downstream, crest"
).format(HEIGHT, args.base, args.crest, FREEBOARD, args.batter, args.element, args.size)

write_mesh(args.out, mesh, comment)

# Se vuelve a leer el archivo, para verificar que quedó bien escrito
check = Mesh()
check.read(args.out)
check.dofs_per_node = 2
elements = create_elements(check)

area = 0.0
for element in elements:
    area = area + element.area()

area_exact = (args.base + args.crest) / 2.0 * HEIGHT

print("Malla:", args.out)
print("  elementos:", check.number_of_elements(), args.element)
print("  nodos    :", check.number_of_nodes(), " (", 2 * check.number_of_nodes(), "grados de libertad )")
print("  área     : {:.4f} m2  (exacta {:.4f} m2)".format(area, area_exact))
print("  lados mojados:", len(check.side_sets["upstream_wet"]))

plot_mesh(
    check, elements,
    side_sets=["upstream_wet", "base"],
    node_sets=["heel", "toe"],
    filename="results/malla_presa_" + args.element.lower() + ".png",
    show=not args.no_show,
    title="Presa de gravedad",
)
