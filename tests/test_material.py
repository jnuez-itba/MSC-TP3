import numpy

from source.materials.linear_elastic import LinearElastic
from source.materials.linear_elastic import PLANE_STRAIN


def test_plane_stress_matrix_is_symmetric():
    material = LinearElastic(21.0e9, 0.2, 2400.0)

    assert numpy.allclose(material.D, material.D.transpose())


def test_uniaxial_stress():
    # Con sigma = [s, 0, 0] tiene que salir exx = s/E, eyy = -nu*s/E, gxy = 0
    E = 21.0e9
    nu = 0.2
    s = 1.0e6

    material = LinearElastic(E, nu)

    strain = numpy.linalg.solve(material.D, numpy.array([s, 0.0, 0.0]))

    assert numpy.allclose(strain, [s / E, -nu * s / E, 0.0])


def test_pure_shear():
    # txy = G * gxy con G = E / (2 (1 + nu))
    E = 21.0e9
    nu = 0.2

    material = LinearElastic(E, nu)

    G = E / (2.0 * (1.0 + nu))

    assert abs(material.D[2][2] - G) < 1.0e-6 * G


def test_plane_strain():
    # Con ezz = 0, para epsilon = [e, 0, 0] la ley de Hooke 3D da
    #   sxx = E (1 - nu) / ((1 + nu)(1 - 2 nu)) * e
    #   syy = E nu / ((1 + nu)(1 - 2 nu)) * e
    # y la tensión fuera del plano szz = nu (sxx + syy) no es nula.
    E = 21.0e9
    nu = 0.2
    e = 1.0e-4

    material = LinearElastic(E, nu, hypothesis=PLANE_STRAIN)

    stress = material.D @ numpy.array([e, 0.0, 0.0])

    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))

    assert numpy.allclose(stress, [c * (1.0 - nu) * e, c * nu * e, 0.0])

    # el módulo de corte es el mismo en las dos hipótesis
    G = E / (2.0 * (1.0 + nu))
    assert abs(material.D[2][2] - G) < 1.0e-6 * G


def test_plane_strain_is_stiffer_than_plane_stress():
    stress = LinearElastic(21.0e9, 0.2)
    strain = LinearElastic(21.0e9, 0.2, hypothesis=PLANE_STRAIN)

    assert strain.D[0][0] > stress.D[0][0]
