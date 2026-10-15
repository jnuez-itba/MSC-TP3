import os

from source.verification.patch import run_patch_test


# rutas relativas a la raíz del repo, así el test anda desde cualquier carpeta
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


TOLERANCE = 1.0e-10


def check(mesh_filename):
    result = run_patch_test(os.path.join(ROOT, mesh_filename))

    assert result["interior_nodes"] > 0
    assert result["displacement_error"] < TOLERANCE
    assert result["stress_error"] < TOLERANCE


def test_patch_tri3():
    check("meshes/patch/patch_tri3.mesh")


def test_patch_quad4():
    check("meshes/patch/patch_quad4.mesh")


def test_patch_tri6():
    check("meshes/patch/patch_tri6.mesh")


def test_patch_quad9():
    check("meshes/patch/patch_quad9.mesh")
