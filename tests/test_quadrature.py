import math

from source.quadrature.gauss import gauss_legendre_1d
from source.quadrature.gauss import gauss_quadrilateral
from source.quadrature.gauss import gauss_triangle


TOLERANCE = 1.0e-12


def exact_1d(p):
    # integral de x^p en [-1, 1]
    if p % 2 == 1:
        return 0.0

    return 2.0 / (p + 1)


def exact_triangle(a, b):
    # integral de x^a y^b en el triángulo de referencia
    return math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)


def test_gauss_1d_exact_up_to_degree_2n_minus_1():
    for n in range(1, 5):
        points, weights = gauss_legendre_1d(n)

        for p in range(2 * n):
            approx = 0.0

            for i in range(n):
                approx = approx + weights[i] * points[i] ** p

            assert abs(approx - exact_1d(p)) < TOLERANCE


def test_gauss_quadrilateral_exact():
    for n in range(1, 5):
        points, weights = gauss_quadrilateral(n)

        for a in range(2 * n):
            for b in range(2 * n):
                approx = 0.0

                for i in range(len(points)):
                    xi = points[i][0]
                    eta = points[i][1]
                    approx = approx + weights[i] * xi ** a * eta ** b

                assert abs(approx - exact_1d(a) * exact_1d(b)) < TOLERANCE


def test_gauss_triangle_exact():
    degrees = {1: 1, 3: 2, 6: 4}

    for n in degrees:
        points, weights = gauss_triangle(n)

        for a in range(degrees[n] + 1):
            for b in range(degrees[n] + 1 - a):
                approx = 0.0

                for i in range(len(points)):
                    x = points[i][0]
                    y = points[i][1]
                    approx = approx + weights[i] * x ** a * y ** b

                assert abs(approx - exact_triangle(a, b)) < 1.0e-10
