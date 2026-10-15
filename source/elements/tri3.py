import numpy

from source.elements.element2d import Element2D
from source.quadrature.gauss import gauss_triangle


class Tri3(Element2D):
    """Triángulo lineal de 3 nodos (CST, deformación constante).

    Elemento de referencia y numeración local (antihoraria):

        eta
         ^
         3
         | \\
         |   \\
         1 --- 2  > xi

        nodo 1: (0, 0)   nodo 2: (1, 0)   nodo 3: (0, 1)

    Funciones de forma:
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta

    B es constante, así que alcanza 1 punto de Gauss.
    """

    number_of_nodes = 3

    # lados: 1-2, 2-3, 3-1
    side_nodes = [(0, 1), (1, 2), (2, 0)]

    def shape_functions(self, xi, eta):
        return numpy.array([
            1.0 - xi - eta,
            xi,
            eta,
        ])

    def shape_derivatives(self, xi, eta):
        return numpy.array([
            [-1.0, 1.0, 0.0],   # dN/dxi
            [-1.0, 0.0, 1.0],   # dN/deta
        ])

    def quadrature(self):
        return gauss_triangle(1)
