import numpy

from source.elements.element2d import Element2D
from source.quadrature.gauss import gauss_quadrilateral


class Quad4(Element2D):
    """Cuadrilátero bilineal de 4 nodos.

    Elemento de referencia [-1, 1] x [-1, 1], numeración antihoraria:

              eta
               ^
        4 -----+----- 3
        |      |      |
        |      +------+--> xi
        |             |
        1 ----------- 2

        nodo 1: (-1, -1)   nodo 2: (1, -1)   nodo 3: (1, 1)   nodo 4: (-1, 1)

    Funciones de forma:
        Ni = 1/4 (1 + xi * xi_i) (1 + eta * eta_i)

    Integración completa: 2 x 2 puntos de Gauss.
    """

    number_of_nodes = 4

    # lados: 1-2, 2-3, 3-4, 4-1
    side_nodes = [(0, 1), (1, 2), (2, 3), (3, 0)]

    reference = "square"

    natural_nodes = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

    # coordenadas naturales de los nodos
    xi_nodes = [-1.0, 1.0, 1.0, -1.0]
    eta_nodes = [-1.0, -1.0, 1.0, 1.0]

    def shape_functions(self, xi, eta):
        N = numpy.zeros(4)

        for i in range(4):
            N[i] = 0.25 * (1.0 + xi * self.xi_nodes[i]) * (1.0 + eta * self.eta_nodes[i])

        return N

    def shape_derivatives(self, xi, eta):
        dN = numpy.zeros((2, 4))

        for i in range(4):
            xi_i = self.xi_nodes[i]
            eta_i = self.eta_nodes[i]

            dN[0][i] = 0.25 * xi_i * (1.0 + eta * eta_i)   # dN/dxi
            dN[1][i] = 0.25 * eta_i * (1.0 + xi * xi_i)    # dN/deta

        return dN

    def quadrature(self):
        return gauss_quadrilateral(2)
