import numpy

from source.elements.element2d import Element2D
from source.quadrature.gauss import gauss_quadrilateral


def lagrange_1d(s):
    """Polinomios de Lagrange cuadráticos en los nodos s = -1, 0, 1."""
    return [
        0.5 * s * (s - 1.0),
        1.0 - s * s,
        0.5 * s * (s + 1.0),
    ]


def lagrange_1d_derivative(s):
    return [
        s - 0.5,
        -2.0 * s,
        s + 0.5,
    ]


class Quad9(Element2D):
    """Cuadrilátero cuadrático lagrangiano de 9 nodos (Q9).

    Elemento de referencia [-1, 1] x [-1, 1]. Numeración local: primero los
    vértices (antihorario), después los nodos de lado y por último el nodo
    central. Es el mismo orden que usa VTK:

              eta
               ^
        4 ---- 7 ---- 3
        |      |      |
        8      9 -----6--> xi
        |             |
        1 ---- 5 ---- 2

        vértices:  1 (-1,-1)  2 ( 1,-1)  3 ( 1, 1)  4 (-1, 1)
        lados:     5 ( 0,-1)  6 ( 1, 0)  7 ( 0, 1)  8 (-1, 0)
        centro:    9 ( 0, 0)

    Las funciones de forma son producto de polinomios de Lagrange 1D:
        Ni(xi, eta) = l_a(xi) * l_b(eta)
    con l_-1(s) = s(s-1)/2, l_0(s) = 1 - s^2, l_1(s) = s(s+1)/2.

    Integración completa: 3 x 3 puntos de Gauss.
    """

    number_of_nodes = 9

    # lados (inicio, fin, medio): 1-2-5, 2-3-6, 3-4-7, 4-1-8
    side_nodes = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]

    # coordenadas naturales de cada nodo, como índice en lagrange_1d:
    # 0 -> s = -1,  1 -> s = 0,  2 -> s = 1
    xi_index = [0, 2, 2, 0, 1, 2, 1, 0, 1]
    eta_index = [0, 0, 2, 2, 0, 1, 2, 1, 1]

    def shape_functions(self, xi, eta):
        l_xi = lagrange_1d(xi)
        l_eta = lagrange_1d(eta)

        N = numpy.zeros(9)

        for i in range(9):
            N[i] = l_xi[self.xi_index[i]] * l_eta[self.eta_index[i]]

        return N

    def shape_derivatives(self, xi, eta):
        l_xi = lagrange_1d(xi)
        l_eta = lagrange_1d(eta)
        dl_xi = lagrange_1d_derivative(xi)
        dl_eta = lagrange_1d_derivative(eta)

        dN = numpy.zeros((2, 9))

        for i in range(9):
            a = self.xi_index[i]
            b = self.eta_index[i]

            dN[0][i] = dl_xi[a] * l_eta[b]    # dN/dxi
            dN[1][i] = l_xi[a] * dl_eta[b]    # dN/deta

        return dN

    def quadrature(self):
        return gauss_quadrilateral(3)
