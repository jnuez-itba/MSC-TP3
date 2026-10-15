import numpy

from source.materials.linear_elastic import LinearElastic


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
