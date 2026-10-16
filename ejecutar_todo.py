"""Ejecuta en orden todos los pasos del script del TP, explicando cada uno.

Pensado para quien entra al repositorio por primera vez: con un solo
comando se verifican los elementos, se resuelve la presa y se generan todos
los resultados en la carpeta results/.

Uso (desde la carpeta del repositorio):

    python ejecutar_todo.py                # paso a paso, con gráficos
    python ejecutar_todo.py --automatico   # todo seguido, sin pausas ni ventanas
    python ejecutar_todo.py --rapido       # saltea los estudios más largos
    python ejecutar_todo.py --paso 5       # solo el paso 5
    python ejecutar_todo.py --desde 5      # del paso 5 en adelante
    python ejecutar_todo.py --sin-color    # sin colores en la terminal

En el modo paso a paso, antes de cada paso se puede ejecutar [Enter],
saltear [s] o salir [q]. Los gráficos se abren en ventanas: hay que
cerrarlas para que el programa siga.

Pasos:
    1. Tests automáticos (pytest)
    2. Patch test de los 4 elementos
    3. Verificación: viga en voladizo
    4. Malla de la presa
    5. Presa con base empotrada: cálculo manual vs FEM
    6. Presa con base Robin (fundación elástica)
    7. Estudio comparativo: elementos, malla, condiciones de borde
    8. Variaciones de geometría para aumentar el factor de seguridad

Autor: Juan Manuel Nuez - 48060
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time


# ----------------------------------------------------------------------
# Opciones
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ejecuta todos los pasos del script del TP3")
parser.add_argument("--automatico", action="store_true",
                    help="correr todo seguido, sin pausas ni ventanas")
parser.add_argument("--rapido", action="store_true",
                    help="saltear el estudio comparativo y el FEM de las geometrías")
parser.add_argument("--paso", type=int, default=None, help="ejecutar solo este paso")
parser.add_argument("--desde", type=int, default=1, help="ejecutar desde este paso")
parser.add_argument("--sin-color", action="store_true", help="no usar colores en la terminal")
args = parser.parse_args()

# Se ejecuta siempre desde la carpeta del repositorio
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PYTHON = sys.executable
INTERACTIVE = not args.automatico
NO_SHOW = ["--no-show"] if args.automatico else []
WIDTH = 76


# ----------------------------------------------------------------------
# Colores (códigos ANSI; os.system("") los activa en la consola de Windows)
# ----------------------------------------------------------------------
USE_COLOR = not args.sin_color
if USE_COLOR and os.name == "nt":
    os.system("")


def color(text, code):
    if not USE_COLOR:
        return text
    return "\033[" + code + "m" + text + "\033[0m"


def bold(text):
    return color(text, "1")


def green(text):
    return color(text, "1;32")


def red(text):
    return color(text, "1;31")


def yellow(text):
    return color(text, "1;33")


def cyan(text):
    return color(text, "1;36")


def line(char="="):
    print(cyan(char * WIDTH))


def centered(text):
    print(bold(text.center(WIDTH).rstrip()))


def ask(prompt):
    """input() que trata el fin de entrada como Enter y Ctrl + C como salir."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""
    except KeyboardInterrupt:
        print("\n" + yellow("Ejecución interrumpida."))
        sys.exit(1)


# ----------------------------------------------------------------------
# Resultados clave de cada paso (se leen de la salida o de los reportes)
# ----------------------------------------------------------------------
def read(filename):
    try:
        with open(filename, encoding="utf-8") as file:
            return file.read()
    except OSError:
        return ""


def find(pattern, text, group=1):
    match = re.search(pattern, text)
    return match.group(group) if match else None


def comma(value):
    """Número con coma decimal, como en el informe."""
    return value.replace(".", ",") if value else "?"


def key_tests(output):
    passed = find(r"(\d+) passed", output)
    failed = find(r"(\d+) failed", output)
    if passed and not failed:
        return passed + " tests pasan"
    if failed:
        return failed + " tests fallan"
    return None


def key_patch(output):
    passed = output.count("PASA")
    return "{} de 8 casos PASAN (4 elementos x 2 versiones), error ~1e-14".format(passed)


def key_beam(output):
    fem = find(r"FEM\s*:\s*([\d.]+) mm", output)
    theory = find(r"Teoría \(flex \+ corte\):\s*([\d.]+) mm", output)
    diff = find(r"Diferencia FEM/teoría:\s*([-\d.]+) %", output)
    if fem and theory:
        return "flecha FEM {} mm vs teoría {} mm ({} %)".format(
            comma(fem), comma(theory), comma(diff))
    return None


def key_mesh(output):
    elements = find(r"elementos:\s*(\d+ \w+)", output)
    dofs = find(r"\(\s*(\d+) grados de libertad", output)
    if elements:
        return "{} elementos, {} grados de libertad".format(elements, dofs)
    return None


def key_fixed(output):
    text = read("results/presa_resultados.txt")
    fs = find(r"FS = Ms / Md\s*=\s*([\d.]+)", text)
    moment = find(r"de las reacciones\s+([-\d.]+)\s+([-\d.]+)", text)
    ux = find(r"Desplazamiento horizontal máximo de la cresta: ([\d.]+) mm", text)
    if fs and moment:
        return ("FS = {} | momento de las reacciones = {} MN·m/m (= -(Ms - Md)) | "
                "u_x cresta = {} mm".format(comma(fs), comma(moment), comma(ux)))
    return None


def key_robin(output):
    text = read("results/presa_resultados_robin.txt")
    ux = find(r"Desplazamiento horizontal máximo de la cresta: ([\d.]+) mm", text)
    s1 = find(r"sigma_1 máxima: ([\d.]+) MPa", text)
    if ux and s1:
        return "u_x cresta = {} mm | sigma_1 máx = {} MPa (se reduce la concentración del talón)".format(
            comma(ux), comma(s1))
    return None


def key_compare(output):
    text = read("results/presa_resultados.txt")
    rows = re.findall(r"^\s+([\d.]+) m\s+\d+\s+([\d.]+)\s+([\d.]+)", text, re.MULTILINE)
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        return ("al refinar de {} a {} m: u_x cresta {} -> {} mm (converge); "
                "sigma_1 talón {} -> {} MPa (singular)".format(
                    comma(first[0]), comma(last[0]), comma(first[1]), comma(last[1]),
                    comma(first[2]), comma(last[2])))
    return None


def key_geometry(output):
    text = read("results/estudio_geometria.txt")
    best = find(r"Base más ancha \(pie aguas abajo\), B = 22 m\s+([\d.]+)", text)
    crest = find(r"Cresta más ancha, c = 7 m\s+([\d.]+)", text)
    if best and crest:
        return "con +19 % de hormigón: base más ancha FS = {} vs cresta más ancha FS = {}".format(
            comma(best), comma(crest))
    return None


# ----------------------------------------------------------------------
# Pasos: (título, explicación, comando, función del resultado clave)
# ----------------------------------------------------------------------
STEPS = [
    ("Tests automáticos",
     "Verifican cuadratura, material, elementos, cargas, mallas y la presa.\n"
     "Tiene que terminar con '81 passed'.",
     [PYTHON, "-m", "pytest", "-q", "tests"], key_tests),

    ("Patch test",
     "Cada elemento (TRI3, QUAD4, TRI6, QUAD9) tiene que reproducir exactamente\n"
     "un estado de tensión constante: A) imponiendo desplazamientos lineales\n"
     "en el borde y B) aplicando una tracción constante. Error ~1e-14 = PASA.",
     [PYTHON, "scripts/patch_test.py"], key_patch),

    ("Verificación: viga en voladizo",
     "Viga de QUAD9 con carga distribuida. Compara la flecha con la teoría de\n"
     "vigas y dibuja la deformada con un factor de escala.",
     [PYTHON, "scripts/ejemplo_viga.py"] + NO_SHOW, key_beam),

    ("Malla de la presa",
     "Genera la sección de la presa (H = 30 m, base 18 m, cresta 3 m) en el\n"
     "formato de malla del TP y la dibuja con sus conjuntos de nodos y lados.\n"
     "(Se escribe en results/ para no modificar las mallas del repositorio.)",
     [PYTHON, "scripts/generar_malla_presa.py", "--out", "results/presa_quad9.mesh"] + NO_SHOW,
     key_mesh),

    ("Presa con base empotrada",
     "Cálculo manual (peso, empuje, Ms, Md, FS, tercio central) y modelo FEM.\n"
     "El momento de las reacciones tiene que ser igual y opuesto a Ms - Md.\n"
     "Dibuja sigma_1 sobre la deformada y la presión en la base.",
     [PYTHON, "scripts/analisis_presa.py"] + NO_SHOW, key_fixed),

    ("Presa con base Robin (bonus)",
     "Misma presa sobre resortes de Winkler (fundación elástica).",
     [PYTHON, "scripts/analisis_presa.py", "--support", "robin"] + NO_SHOW, key_robin),

    ("Estudio comparativo",
     "Compara los 4 elementos, refina la malla, compara empotrada / rodillo /\n"
     "Robin y deformación plana vs tensión plana. Abre varias ventanas y\n"
     "tarda unos segundos más que los otros pasos.",
     [PYTHON, "scripts/analisis_presa.py", "--comparar"] + NO_SHOW, key_compare),

    ("Variaciones de geometría",
     "Cuatro familias de geometrías: base más ancha, talón extendido, cresta\n"
     "más ancha y talud. Factor de seguridad al vuelco vs área de hormigón.",
     [PYTHON, "scripts/estudio_geometria.py"] + (["--sin-fem"] if args.rapido else []) + NO_SHOW,
     key_geometry),
]

if args.rapido:
    STEPS[6] = None          # el estudio comparativo es el más largo

# Scripts que abren ventanas con gráficos
GRAPHIC_SCRIPTS = ["scripts/ejemplo_viga.py", "scripts/generar_malla_presa.py",
                   "scripts/analisis_presa.py", "scripts/estudio_geometria.py"]


# ----------------------------------------------------------------------
# Librerías necesarias
# ----------------------------------------------------------------------
def check_libraries():
    required = ["numpy", "matplotlib", "meshio"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    has_pytest = importlib.util.find_spec("pytest") is not None

    if missing:
        print(red("Faltan librerías: " + ", ".join(missing)))
        print("Instalalas con:")
        print(bold("    pip install -r requirements.txt"))
        sys.exit(1)

    if not has_pytest:
        print(yellow("Aviso: pytest no está instalado; el paso 1 se va a saltear."))
        print("Para incluirlo: " + bold("pip install -r requirements.txt"))
        print()

    return has_pytest


# ----------------------------------------------------------------------
# Saludos
# ----------------------------------------------------------------------
def welcome(steps_to_run):
    print()
    line()
    centered("TP3 - MECÁNICA DEL SÓLIDO COMPUTACIONAL (30.53) - ITBA")
    centered("Elementos finitos en estado plano: presa de gravedad")
    line()
    print()
    print("¡Bienvenido! Este programa ejecuta en orden todos los pasos del script del TP:")
    print()
    for number, step in enumerate(STEPS, start=1):
        if step is None:
            continue
        mark = "" if number in steps_to_run else "   (no se ejecuta)"
        print("  {}. {}{}".format(number, step[0], mark))
    print()
    if INTERACTIVE:
        print("Antes de cada paso se explica qué hace. En cada pausa podés ejecutar")
        print("[Enter], saltear [s] o salir [q]. Los gráficos se abren en ventanas:")
        print("cerralas para continuar.")
    print("Los resultados se guardan en la carpeta results/.")
    print()
    if INTERACTIVE:
        answer = ask(bold("Presioná Enter para comenzar (q para salir)... "))
        if answer == "q":
            print("Hasta la próxima.")
            sys.exit(0)


def farewell(complete):
    print()
    line()
    if complete:
        centered("Listo! Se ejecutó todo el script del TP.")
    else:
        centered("Listo! Se ejecutaron los pasos elegidos del script del TP.")
    line()
    print()
    print("Se verificaron los 4 elementos y se resolvió la presa de gravedad con tres")
    print("condiciones de borde en la base. Los reportes y las figuras quedaron en la")
    print("carpeta results/, y el análisis completo está en informe/informe_TP3_final.pdf.")
    print()
    print("¡Gracias por revisar el trabajo!")
    print()
    print(bold("Juan Manuel Nuez - 48060".rjust(WIDTH - 2)))
    line()


def offer_report():
    pdf = os.path.join("informe", "informe_TP3_final.pdf")
    if not INTERACTIVE or not os.path.exists(pdf):
        return
    answer = ask("\n¿Abrir el informe en PDF? [s/n] ")
    if answer not in ("s", "si", "sí"):
        return
    try:
        if os.name == "nt":
            os.startfile(pdf)
        elif sys.platform == "darwin":
            subprocess.call(["open", pdf])
        else:
            subprocess.call(["xdg-open", pdf])
    except OSError:
        print("No se pudo abrir automáticamente; está en " + pdf)


# ----------------------------------------------------------------------
# Ejecución de un paso, mostrando la salida en vivo y guardándola
# ----------------------------------------------------------------------
def run(command):
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=environment, encoding="utf-8", errors="replace")
    output = []
    for text in process.stdout:
        print(text, end="")
        output.append(text)
    process.wait()
    return process.returncode, "".join(output)


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
has_pytest = check_libraries()

if args.paso is not None:
    steps_to_run = [args.paso]
else:
    steps_to_run = list(range(args.desde, len(STEPS) + 1))

steps_to_run = [n for n in steps_to_run if 1 <= n <= len(STEPS) and STEPS[n - 1] is not None]
if not steps_to_run:
    print(red("No hay pasos para ejecutar. Los pasos van de 1 a {}.".format(len(STEPS))))
    sys.exit(1)

os.makedirs("results", exist_ok=True)
welcome(steps_to_run)

summary = []
stopped = False
start_all = time.time()

for number in steps_to_run:
    title, explanation, command, key_result = STEPS[number - 1]

    print()
    line()
    print(bold("PASO {} de {}: {}".format(number, len(STEPS), title)))
    line()
    print(explanation)
    print("\n> " + " ".join("python" if c == PYTHON else c for c in command))

    if number == 1 and not has_pytest:
        print(yellow("\nSalteado: pytest no está instalado."))
        summary.append((number, title, "SALTEADO", 0.0))
        continue

    if INTERACTIVE:
        answer = ask("\n[Enter] ejecutar   [s] saltear este paso   [q] salir: ")
        if answer == "q":
            print(yellow("Ejecución interrumpida por el usuario."))
            stopped = True
            break
        if answer == "s":
            summary.append((number, title, "SALTEADO", 0.0))
            continue
        if command[1] in GRAPHIC_SCRIPTS:
            print("(se abren ventanas con gráficos: cerralas para continuar)")
    print()

    start = time.time()
    code, output = run(command)
    elapsed = time.time() - start

    if code == 0:
        result = key_result(output)
        if result:
            print("\n" + green("→ Resultado: ") + result)
        summary.append((number, title, "OK", elapsed))
    else:
        print("\n" + red("→ El paso terminó con errores (revisar la salida de arriba)."))
        summary.append((number, title, "ERROR", elapsed))


# ----------------------------------------------------------------------
# Resumen y despedida
# ----------------------------------------------------------------------
print()
line()
print(bold("RESUMEN"))
line()
for number, title, state, elapsed in summary:
    if state == "OK":
        state_text = green("{:>8}".format(state))
    elif state == "ERROR":
        state_text = red("{:>8}".format(state))
    else:
        state_text = yellow("{:>8}".format(state))
    print("  {}. {:<44} {}   {:6.1f} s".format(number, title, state_text, elapsed))
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

complete = (not stopped and len(steps_to_run) == sum(1 for s in STEPS if s is not None)
            and all(state == "OK" for _, _, state, _ in summary))
farewell(complete)
offer_report()

sys.exit(0 if all(state != "ERROR" for _, _, state, _ in summary) else 1)
