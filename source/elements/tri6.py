import numpy

from source.elements.element2d import Element2D
from source.quadrature.gauss import gauss_triangle


class Tri6(Element2D):
    """Triángulo cuadrático de 6 nodos (LST, deformación lineal).

    Elemento de referencia y numeración local (antihoraria, primero los
    vértices y después los nodos de lado; es el mismo orden que usa VTK):

        eta
         ^
         3
         | \\
         6   5
         |     \\
         1 --4-- 2  > xi

        vértices:  1 (0, 0)    2 (1, 0)    3 (0, 1)
        lados:     4 (1/2, 0)  5 (1/2, 1/2)  6 (0, 1/2)

    Con las coordenadas de área L1 = 1 - xi - eta, L2 = xi, L3 = eta:
        vértices:  Ni = Li (2 Li - 1)
        lados:     N4 = 4 L1 L2,  N5 = 4 L2 L3,  N6 = 4 L3 L1

    B es lineal y B^T D B es de grado 2, así que alcanzan 3 puntos de Gauss.
    """

    number_of_nodes = 6

    # lados (inicio, fin, medio): 1-2-4, 2-3-5, 3-1-6
    side_nodes = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]

    natural_nodes = [
        (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        (0.5, 0.0), (0.5, 0.5), (0.0, 0.5),
    ]

    def shape_functions(self, xi, eta):
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta

        return numpy.array([
            L1 * (2.0 * L1 - 1.0),
            L2 * (2.0 * L2 - 1.0),
            L3 * (2.0 * L3 - 1.0),
            4.0 * L1 * L2,
            4.0 * L2 * L3,
            4.0 * L3 * L1,
        ])

    def shape_derivatives(self, xi, eta):
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta

        # Derivadas de las coordenadas de área:
        #   dL1/dxi = -1, dL2/dxi = 1, dL3/dxi = 0
        #   dL1/deta = -1, dL2/deta = 0, dL3/deta = 1
        dN = numpy.zeros((2, 6))

        dL = [
            [-1.0, 1.0, 0.0],   # d/dxi
            [-1.0, 0.0, 1.0],   # d/deta
        ]

        for k in range(2):
            dL1 = dL[k][0]
            dL2 = dL[k][1]
            dL3 = dL[k][2]

            dN[k][0] = (4.0 * L1 - 1.0) * dL1
            dN[k][1] = (4.0 * L2 - 1.0) * dL2
            dN[k][2] = (4.0 * L3 - 1.0) * dL3
            dN[k][3] = 4.0 * (dL1 * L2 + L1 * dL2)
            dN[k][4] = 4.0 * (dL2 * L3 + L2 * dL3)
            dN[k][5] = 4.0 * (dL3 * L1 + L3 * dL1)

        return dN

    def quadrature(self):
        return gauss_triangle(3)
