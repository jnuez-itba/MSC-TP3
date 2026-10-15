"""Ejemplo: viga en voladizo con carga distribuida.

Viga de 10 m x 1 m (Q9), empotrada en x = 0, con una presión q sobre la cara
superior. Compara la flecha en la punta con la teoría de vigas y muestra la
deformada.

Uso:
    python ejemplo_viga.py                        # sigma_1, escala automática
    python ejemplo_viga.py --scale 50             # factor de escala 50
    python ejemplo_viga.py --field von_mises      # otro campo
    python ejemplo_viga.py --no-show              # solo guarda los archivos

Campos: sigma_1, sigma_2, sigma_xx, sigma_yy, tau_xy, von_mises, displacement

Resultados en la carpeta results/:
    viga.vtu          para abrir en ParaView
    viga_<campo>.png  la figura
"""

import argparse

import numpy

from source.mesh.mesh import Mesh
from source.elements.factory import create_elements
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import fix_node_set
from source.systems.loads import add_side_pressure
from source.solvers.linear import solve
from source.output.paraview import write_solid_results
from source.output.plot import plot_deformed
from source.output.plot import FIELDS


parser = argparse.ArgumentParser(description="Viga en voladizo")
parser.add_argument("--scale", type=float, default=None,
                    help="factor de escala de la deformada (por defecto automático)")
parser.add_argument("--field", default="sigma_1", choices=list(FIELDS.keys()),
                    help="campo con el que se colorea la deformada")
parser.add_argument("--no-show", action="store_true",
                    help="no abrir la ventana del gráfico")
args = parser.parse_args()


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
q = 1.0e5          # presión sobre la cara superior [Pa] = [N/m por m de espesor]
L = 10.0           # largo [m]
h = 1.0            # altura [m]


# ----------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------
mesh = Mesh()
mesh.read("meshes/ejemplos/viga_q9.mesh")
mesh.dofs_per_node = 2

elements = create_elements(mesh)

K = DenseMatrix(mesh.number_of_dofs())
for element in elements:
    K.scatter_local_to_global(element)

f = numpy.zeros(mesh.number_of_dofs())
add_side_pressure(f, mesh, elements, "top", lambda x, y: q)

fix_node_set(K, f, mesh, "left", (0, 1))

u = solve(K, f)


# ----------------------------------------------------------------------
# Comparación con teoría de vigas (Timoshenko: flexión + corte)
# ----------------------------------------------------------------------
material = mesh.element_material(0)
t = mesh.thickness
E = material.E
G = E / (2.0 * (1.0 + material.nu))
I = t * h ** 3 / 12.0
A = t * h
w = q * t                            # carga por unidad de largo [N/m]

bending = w * L ** 4 / (8.0 * E * I)
shear = w * L ** 2 / (2.0 * (5.0 / 6.0) * G * A)

# nodo de la punta, a media altura
tip = None
for node in mesh.nodes:
    if abs(node.x - L) < 1.0e-9 and abs(node.y - h / 2.0) < 1.0e-9:
        tip = node.global_id

v_tip = -u[mesh.node_global_dofs(tip)[1]]

print("Flecha en la punta")
print("  FEM                  : {:.4f} mm".format(v_tip * 1.0e3))
print("  Teoría (flexión)     : {:.4f} mm".format(bending * 1.0e3))
print("  Teoría (flex + corte): {:.4f} mm".format((bending + shear) * 1.0e3))
print("  Diferencia FEM/teoría: {:.2f} %".format(
    100.0 * (v_tip - bending - shear) / (bending + shear)))


# ----------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------
write_solid_results("results/viga.vtu", mesh, elements, u)

scale = plot_deformed(
    mesh, elements, u,
    field=args.field,
    scale=args.scale,
    filename="results/viga_" + args.field + ".png",
    show=not args.no_show,
    title="Viga en voladizo",
)

print()
print("Factor de escala usado:", scale)
print("Guardado: results/viga.vtu y results/viga_" + args.field + ".png")
