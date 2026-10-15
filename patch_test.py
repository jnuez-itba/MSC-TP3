"""Corre el patch test para cada tipo de elemento e imprime los resultados.

  A) por desplazamientos: campo lineal impuesto en todo el borde
  B) por fuerzas: tracción de tensión constante en el borde, apoyos mínimos

Uso:
    python patch_test.py
"""

from source.verification.patch import run_patch_test
from source.verification.patch import run_patch_test_forces


MESHES = [
    ("TRI3", "meshes/patch/patch_tri3.mesh"),
    ("QUAD4", "meshes/patch/patch_quad4.mesh"),
    ("TRI6", "meshes/patch/patch_tri6.mesh"),
    ("QUAD9", "meshes/patch/patch_quad9.mesh"),
]

TOLERANCE = 1.0e-10

all_passed = True


print("A) Patch test por desplazamientos (campo lineal impuesto en el borde)")
print()
print("{:<8}{:>10}{:>12}{:>18}{:>18}{:>10}".format(
    "Tipo", "Elementos", "Interiores", "Error u (rel)", "Error sigma (rel)", ""))

for element_type, filename in MESHES:
    result = run_patch_test(filename)

    passed = (
        result["displacement_error"] < TOLERANCE
        and result["stress_error"] < TOLERANCE
    )

    all_passed = all_passed and passed

    print("{:<8}{:>10}{:>12}{:>18.2e}{:>18.2e}{:>10}".format(
        element_type,
        result["elements"],
        result["interior_nodes"],
        result["displacement_error"],
        result["stress_error"],
        "PASA" if passed else "FALLA",
    ))

print()
print("Tensión exacta [sxx, syy, txy] (Pa):", result["stress_exact"])
print()
print()
print("B) Patch test por fuerzas (tracción constante en el borde, apoyos mínimos)")
print()
print("{:<8}{:>10}{:>18}{:>10}".format("Tipo", "Elementos", "Error sigma (rel)", ""))

for element_type, filename in MESHES:
    result = run_patch_test_forces(filename)

    passed = result["stress_error"] < TOLERANCE

    all_passed = all_passed and passed

    print("{:<8}{:>10}{:>18.2e}{:>10}".format(
        element_type,
        result["elements"],
        result["stress_error"],
        "PASA" if passed else "FALLA",
    ))

if not all_passed:
    raise SystemExit(1)
