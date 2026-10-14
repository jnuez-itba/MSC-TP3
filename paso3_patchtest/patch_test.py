"""Corre el patch test para cada tipo de elemento e imprime los resultados.

Uso:
    python patch_test.py
"""

from source.verification.patch import run_patch_test


MESHES = [
    ("TRI3", "meshes/patch/patch_tri3.mesh"),
    ("QUAD4", "meshes/patch/patch_quad4.mesh"),
]

TOLERANCE = 1.0e-10


print("Patch test (desplazamientos lineales impuestos en el borde)")
print()
print("{:<8}{:>10}{:>12}{:>18}{:>18}{:>10}".format(
    "Tipo", "Elementos", "Interiores", "Error u (rel)", "Error sigma (rel)", ""))

all_passed = True

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

if not all_passed:
    raise SystemExit(1)
