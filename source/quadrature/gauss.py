"""Reglas de integración numérica.

Cada función devuelve (points, weights):
    points   lista de coordenadas naturales [xi] o [xi, eta]
    weights  lista de pesos

La integral se aproxima como  sum_i f(points[i]) * weights[i].
"""

import math


def gauss_legendre_1d(n):
    """Gauss-Legendre de n puntos en [-1, 1].

    Integra exactamente polinomios de grado 2n - 1.
    """
    if n == 1:
        points = [0.0]
        weights = [2.0]

    elif n == 2:
        a = 1.0 / math.sqrt(3.0)
        points = [-a, a]
        weights = [1.0, 1.0]

    elif n == 3:
        a = math.sqrt(3.0 / 5.0)
        points = [-a, 0.0, a]
        weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

    elif n == 4:
        a = math.sqrt(3.0 / 7.0 - 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
        b = math.sqrt(3.0 / 7.0 + 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
        wa = (18.0 + math.sqrt(30.0)) / 36.0
        wb = (18.0 - math.sqrt(30.0)) / 36.0
        points = [-b, -a, a, b]
        weights = [wb, wa, wa, wb]

    else:
        raise ValueError("Unsupported number of Gauss points")

    return points, weights


def gauss_quadrilateral(n):
    """Regla producto tensorial de n x n puntos en [-1, 1] x [-1, 1].

    Integra exactamente polinomios de grado 2n - 1 en cada variable.
    """
    points_1d, weights_1d = gauss_legendre_1d(n)

    points = []
    weights = []

    for j in range(n):
        for i in range(n):
            points.append([points_1d[i], points_1d[j]])
            weights.append(weights_1d[i] * weights_1d[j])

    return points, weights


def gauss_triangle(n):
    """Regla de n puntos en el triángulo de referencia (0,0), (1,0), (0,1).

    El área del triángulo de referencia es 1/2, por eso los pesos suman 1/2.

    n = 1  grado 1 (centroide)
    n = 3  grado 2
    n = 6  grado 4 (Dunavant)
    """
    if n == 1:
        points = [[1.0 / 3.0, 1.0 / 3.0]]
        weights = [0.5]

    elif n == 3:
        a = 1.0 / 6.0
        b = 2.0 / 3.0
        points = [[a, a], [b, a], [a, b]]
        weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]

    elif n == 6:
        a = 0.445948490915965
        b = 0.091576213509771
        wa = 0.223381589678011 / 2.0
        wb = 0.109951743655322 / 2.0
        points = [
            [a, a], [1.0 - 2.0 * a, a], [a, 1.0 - 2.0 * a],
            [b, b], [1.0 - 2.0 * b, b], [b, 1.0 - 2.0 * b],
        ]
        weights = [wa, wa, wa, wb, wb, wb]

    else:
        raise ValueError("Unsupported number of triangle points")

    return points, weights
