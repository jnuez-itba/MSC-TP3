"""Ejecuta todo el TP en orden, explicando cada paso.

Pensado para quien entra al repositorio por primera vez: con un solo
comando se verifican los elementos, se resuelve la presa y se generan todos
los resultados en la carpeta results/.

Uso (desde la carpeta del repositorio):

    python ejecutar_todo.py              # todo, sin abrir ventanas
    python ejecutar_todo.py --mostrar    # abre cada gráfico (cerrarlo para seguir)
    python ejecutar_todo.py --rapido     # saltea los estudios más largos

Pasos:
    1. Tests automáticos (pytest)
    2. Patch test de los 4 elementos
    3. Ejemplo de verificación: viga en voladizo
    4. Malla de la presa
    5. Presa con base empotrada: cálculo manual vs FEM
    6. Presa con base Robin (fundación elástica)
    7. Estudio comparativo: elementos, malla, condiciones de borde
    8. Variaciones de geometría para aumentar el factor de seguridad
"""

import argparse
import os
import subprocess
import sys
import time


parser = argparse.ArgumentParser(description="Ejecuta todos los pasos del TP3")
parser.add_argument("--mostrar", action="store_true",
                    help="abrir las ventanas de los gráficos")
parser.add_argument("--rapido", action="store_true",
                    help="saltear el estudio comparativo y el FEM de las geometrías")
args = parser.parse_args()

# Se ejecuta siempre desde la carpeta del repositorio
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

PYTHON = sys.executable
NO_SHOW = [] if args.mostrar else ["--no-show"]


STEPS = [
    ("Tests automáticos",
     "Verifican cuadratura, material, elementos, cargas, mallas y la presa.\n"
     "Tiene que terminar con '81 passed'.",
     [PYTHON, "-m", "pytest", "-q"]),

    ("Patch test",
     "Cada elemento (TRI3, QUAD4, TRI6, QUAD9) tiene que reproducir exactamente\n"
     "un estado de tensión constante: A) imponiendo desplazamientos lineales\n"
     "en el borde y B) aplicando una tracción constante. Error ~1e-14 = PASA.",
     [PYTHON, "patch_test.py"]),

    ("Ejemplo de verificación: viga en voladizo",
     "Viga de QUAD9 con carga distribuida. Compara la flecha con la teoría de\n"
     "vigas y dibuja la deformada con un factor de escala.",
     [PYTHON, "ejemplo_viga.py"] + NO_SHOW),

    ("Malla de la presa",
     "Genera la sección de la presa (H = 30 m, base 18 m, cresta 3 m) en el\n"
     "formato de malla del TP y la dibuja con sus conjuntos de nodos y lados.\n"
     "(Se escribe en results/ para no modificar las mallas del repositorio.)",
     [PYTHON, "generar_malla_presa.py", "--out", "results/presa_quad9.mesh"] + NO_SHOW),

    ("Presa con base empotrada",
     "Cálculo manual (peso, empuje, Ms, Md, FS, tercio central) y modelo FEM.\n"
     "El momento de las reacciones tiene que ser igual y opuesto a Ms - Md.\n"
     "Dibuja sigma_1 sobre la deformada.",
     [PYTHON, "analisis_presa.py"] + NO_SHOW),

    ("Presa con base Robin (bonus)",
     "Misma presa sobre resortes de Winkler (fundación elástica).",
     [PYTHON, "analisis_presa.py", "--support", "robin"] + NO_SHOW),
]

if not args.rapido:
    STEPS.append(
        ("Estudio comparativo",
         "Compara los 4 elementos, refina la malla, compara empotrada / rodillo /\n"
         "Robin y deformación plana vs tensión plana. Puede tardar un poco.",
         [PYTHON, "analisis_presa.py", "--comparar"] + NO_SHOW))

STEPS.append(
    ("Variaciones de geometría",
     "Cuatro familias de geometrías: base más ancha, talón extendido, cresta\n"
     "más ancha y talud. Factor de seguridad al vuelco vs área de hormigón.",
     [PYTHON, "estudio_geometria.py"] + (["--sin-fem"] if args.rapido else []) + NO_SHOW))


def banner(text):
    print()
    print("=" * 76)
    print(text)
    print("=" * 76)


results = []
start_all = time.time()

for k, (title, explanation, command) in enumerate(STEPS, start=1):
    banner("PASO {} de {}: {}".format(k, len(STEPS), title))
    print(explanation)
    print("\n> " + " ".join(os.path.basename(c) if c == PYTHON else c for c in command))
    print()

    start = time.time()
    code = subprocess.call(command)
    elapsed = time.time() - start

    if code != 0 and command[1:3] == ["-m", "pytest"]:
        print("\n(pytest no está instalado o algún test falló: pip install -r requirements.txt)")

    results.append((title, code == 0, elapsed))


banner("RESUMEN")
for title, ok, elapsed in results:
    print("  {:<45} {:>6}   {:6.1f} s".format(title, "OK" if ok else "ERROR", elapsed))
print("\n  Tiempo total: {:.1f} s".format(time.time() - start_all))

print("\nResultados en la carpeta results/:")
for name in sorted(os.listdir("results")):
    print("  " + name)

print("""
Cómo leerlos:
  presa_resultados.txt          reporte principal (base empotrada + estudio comparativo)
  presa_resultados_robin.txt    reporte con base Robin
  estudio_geometria.txt         variaciones de geometría
  presa_sigma_1_*.png           sigma_1 sobre la deformada
  presa_base_*.png              presión en la base: FEM vs ley trapecial
  estudio_*.png                 factor de seguridad vs área y secciones
  *.vtu                         para abrir en ParaView (filtro Warp By Vector)""")

sys.exit(0 if all(ok for _, ok, _ in results) else 1)
