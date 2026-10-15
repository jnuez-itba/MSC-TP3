import math

import numpy

from source.quadrature.gauss import gauss_legendre_1d


class Element2D:
    """Clase base para elementos isoparamétricos de estado plano.

    Cada subclase define:
        number_of_nodes            cantidad de nodos del elemento
        side_nodes                 nodos locales de cada lado: (inicio, fin)
                                   o (inicio, fin, medio)
        natural_nodes              coordenadas naturales (xi, eta) de cada nodo
        reference                  "triangle" o "square" (elemento de referencia)
        shape_functions(xi, eta)   vector N con las funciones de forma
        shape_derivatives(xi, eta) matriz 2 x n con [dN/dxi; dN/deta]
        quadrature()               puntos y pesos de Gauss

    Con eso, esta clase calcula todo lo demás: jacobiano, matriz B, rigidez,
    deformaciones y tensiones.

    Convención de grados de libertad del elemento:
        [u1, v1, u2, v2, ..., un, vn]
    Convención de Voigt:
        epsilon = [exx, eyy, gxy]   sigma = [sxx, syy, txy]
    """

    dofs_per_node = 2
    number_of_nodes = 0
    side_nodes = []
    natural_nodes = []
    reference = ""

    def __init__(self, global_id, node_global_ids, mesh, material, thickness):
        if len(node_global_ids) != self.number_of_nodes:
            raise ValueError(
                "Expected " + str(self.number_of_nodes) + " nodes, got "
                + str(len(node_global_ids))
            )

        if thickness <= 0.0:
            raise ValueError("Expected positive thickness")

        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.thickness = thickness

        # Coordenadas nodales, una fila por nodo: [[x1, y1], [x2, y2], ...]
        self.coordinates = numpy.zeros((self.number_of_nodes, 2))

        for i in range(self.number_of_nodes):
            node = mesh.node(node_global_ids[i])
            self.coordinates[i][0] = node.x
            self.coordinates[i][1] = node.y

        self.Ke = self.compute_stiffness_matrix()

    # ------------------------------------------------------------------
    # A definir en cada subclase
    # ------------------------------------------------------------------

    def shape_functions(self, xi, eta):
        raise NotImplementedError

    def shape_derivatives(self, xi, eta):
        raise NotImplementedError

    def quadrature(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Geometría
    # ------------------------------------------------------------------

    def jacobian(self, xi, eta):
        """Jacobiano de la transformación (xi, eta) -> (x, y).

            J = | dx/dxi   dy/dxi  |  = dN_natural @ coordinates
                | dx/deta  dy/deta |
        """
        dN_natural = self.shape_derivatives(xi, eta)

        return dN_natural @ self.coordinates

    def shape_derivatives_xy(self, xi, eta):
        """Derivadas de las funciones de forma respecto de x, y.

        Por regla de la cadena:  dN_natural = J @ dN_xy
        entonces                 dN_xy = J^-1 @ dN_natural

        Devuelve (dN_xy, detJ), con dN_xy de 2 x n: [dN/dx; dN/dy].
        """
        dN_natural = self.shape_derivatives(xi, eta)
        J = dN_natural @ self.coordinates
        detJ = numpy.linalg.det(J)

        if detJ <= 0.0:
            raise ValueError(
                "Element " + str(self.global_id + 1)
                + ": non-positive Jacobian (check counterclockwise node order)"
            )

        dN_xy = numpy.linalg.solve(J, dN_natural)

        return dN_xy, detJ

    def area(self):
        A = 0.0
        points, weights = self.quadrature()

        for p in range(len(points)):
            J = self.jacobian(points[p][0], points[p][1])
            A = A + numpy.linalg.det(J) * weights[p]

        return A

    # ------------------------------------------------------------------
    # Rigidez
    # ------------------------------------------------------------------

    def B_matrix(self, xi, eta):
        """Matriz deformación-desplazamiento B (3 x 2n): epsilon = B @ ue.

            exx = du/dx          fila 0: [dN1/dx   0      dN2/dx   0     ...]
            eyy = dv/dy          fila 1: [0        dN1/dy 0        dN2/dy...]
            gxy = du/dy + dv/dx  fila 2: [dN1/dy   dN1/dx dN2/dy   dN2/dx...]

        Devuelve (B, detJ).
        """
        dN_xy, detJ = self.shape_derivatives_xy(xi, eta)

        n = self.number_of_nodes
        B = numpy.zeros((3, 2 * n))

        for i in range(n):
            dNdx = dN_xy[0][i]
            dNdy = dN_xy[1][i]

            B[0][2 * i] = dNdx
            B[1][2 * i + 1] = dNdy
            B[2][2 * i] = dNdy
            B[2][2 * i + 1] = dNdx

        return B, detJ

    def compute_stiffness_matrix(self):
        """Ke = integral( B^T D B t dA ) ~= sum_p B^T D B t detJ w_p"""
        n_dofs = 2 * self.number_of_nodes
        Ke = numpy.zeros((n_dofs, n_dofs))

        D = self.material.D
        t = self.thickness

        points, weights = self.quadrature()

        for p in range(len(points)):
            B, detJ = self.B_matrix(points[p][0], points[p][1])
            Ke = Ke + B.transpose() @ D @ B * t * detJ * weights[p]

        return Ke

    def stiffness_matrix(self):
        return self.Ke

    # ------------------------------------------------------------------
    # Grados de libertad (misma interfaz que Bar2D)
    # ------------------------------------------------------------------

    def global_dofs(self):
        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def element_displacements(self, u):
        dofs = self.global_dofs()

        ue = numpy.zeros(len(dofs))

        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        return ue

    # ------------------------------------------------------------------
    # Post-proceso
    # ------------------------------------------------------------------

    def strain(self, u, xi, eta):
        """[exx, eyy, gxy] en el punto (xi, eta)."""
        B, detJ = self.B_matrix(xi, eta)

        return B @ self.element_displacements(u)

    def stress(self, u, xi, eta):
        """[sxx, syy, txy] en el punto (xi, eta)."""
        return self.material.D @ self.strain(u, xi, eta)

    def stress_at_gauss_points(self, u):
        points, weights = self.quadrature()

        stresses = []

        for p in range(len(points)):
            stresses.append(self.stress(u, points[p][0], points[p][1]))

        return stresses

    def stress_at_nodes(self, u):
        """Tensiones [sxx, syy, txy] evaluadas en cada nodo del elemento.

        Se evalúan las funciones de forma en las coordenadas naturales de
        cada nodo. Como la tensión es discontinua entre elementos, en la
        malla se promedian los valores de los elementos que comparten un nodo
        (ver source/output/postprocess.py).
        """
        stresses = []

        for xi, eta in self.natural_nodes:
            stresses.append(self.stress(u, xi, eta))

        return numpy.array(stresses)

    def contains_natural(self, xi, eta, tolerance=1.0e-9):
        """True si (xi, eta) está dentro del elemento de referencia."""
        if self.reference == "triangle":
            return (xi >= -tolerance and eta >= -tolerance
                    and xi + eta <= 1.0 + tolerance)

        return abs(xi) <= 1.0 + tolerance and abs(eta) <= 1.0 + tolerance

    def natural_coordinates(self, x, y, iterations=20):
        """Inversa del mapeo isoparamétrico: (xi, eta) del punto (x, y).

        Se resuelve x(xi, eta) = x_objetivo con Newton-Raphson. Devuelve None
        si no converge. No verifica que el punto esté dentro del elemento
        (para eso, contains_natural).
        """
        if self.reference == "triangle":
            natural = numpy.array([1.0 / 3.0, 1.0 / 3.0])
        else:
            natural = numpy.array([0.0, 0.0])

        target = numpy.array([x, y])
        size = numpy.ptp(self.coordinates, axis=0).max()

        for k in range(iterations):
            residual = self.physical_coordinates(natural[0], natural[1]) - target

            if numpy.abs(residual).max() < 1.0e-12 * size:
                return natural

            # d(x, y)/d(xi, eta) = J^T
            J = self.jacobian(natural[0], natural[1])
            natural = natural - numpy.linalg.solve(J.transpose(), residual)

        return None

    def physical_coordinates(self, xi, eta):
        """(x, y) del punto de coordenadas naturales (xi, eta)."""
        N = self.shape_functions(xi, eta)

        return N @ self.coordinates

    # ------------------------------------------------------------------
    # Cargas
    # ------------------------------------------------------------------

    def body_force_vector(self, bx, by):
        """Fuerzas nodales equivalentes a una fuerza de volumen constante
        b = (bx, by) [N/m3]:

            fe = integral( N^T b t dA )

        Para el peso propio: b = (0, -rho g).
        """
        fe = numpy.zeros(2 * self.number_of_nodes)
        t = self.thickness

        points, weights = self.quadrature()

        for p in range(len(points)):
            xi = points[p][0]
            eta = points[p][1]

            N = self.shape_functions(xi, eta)
            detJ = numpy.linalg.det(self.jacobian(xi, eta))

            for i in range(self.number_of_nodes):
                fe[2 * i] = fe[2 * i] + N[i] * bx * t * detJ * weights[p]
                fe[2 * i + 1] = fe[2 * i + 1] + N[i] * by * t * detJ * weights[p]

        return fe

    def side_load_vector(self, side, traction_function):
        """Fuerzas nodales equivalentes a una tracción sobre un lado:

            fe = integral_lado( N^T t(x, y) espesor ds )

        side               número de lado (desde 0). El lado k va del vértice
                           k al vértice k + 1 (ver side_nodes de cada elemento).
        traction_function  función (x, y, normal) -> (tx, ty) [N/m2], donde
                           normal es la normal unitaria saliente del lado.

        Se integra con 3 puntos de Gauss sobre el lado: es exacto para
        presiones que varían linealmente (como la hidrostática), tanto en
        lados lineales como cuadráticos.
        """
        side_nodes = list(self.side_nodes[side])
        side_coordinates = self.coordinates[side_nodes]

        fe = numpy.zeros(2 * self.number_of_nodes)
        t = self.thickness

        points, weights = gauss_legendre_1d(3)

        for p in range(len(points)):
            N, dN = side_shape_functions(len(side_nodes), points[p])

            x, y = N @ side_coordinates
            dx, dy = dN @ side_coordinates
            ds = math.sqrt(dx * dx + dy * dy)

            # Con los vértices en sentido antihorario, la normal saliente
            # es la tangente rotada 90 grados en sentido horario.
            normal = numpy.array([dy / ds, -dx / ds])

            tx, ty = traction_function(x, y, normal)

            for a in range(len(side_nodes)):
                i = side_nodes[a]
                fe[2 * i] = fe[2 * i] + N[a] * tx * t * ds * weights[p]
                fe[2 * i + 1] = fe[2 * i + 1] + N[a] * ty * t * ds * weights[p]

        return fe

    def side_coordinates(self, side):
        return self.coordinates[list(self.side_nodes[side])]


def side_shape_functions(n_nodes, s):
    """Funciones de forma 1D sobre un lado, en s en [-1, 1].

    2 nodos (lado lineal):     [inicio, fin]
    3 nodos (lado cuadrático): [inicio, fin, medio]

    Devuelve (N, dN/ds).
    """
    if n_nodes == 2:
        N = numpy.array([(1.0 - s) / 2.0, (1.0 + s) / 2.0])
        dN = numpy.array([-0.5, 0.5])

    elif n_nodes == 3:
        N = numpy.array([s * (s - 1.0) / 2.0, s * (s + 1.0) / 2.0, 1.0 - s * s])
        dN = numpy.array([s - 0.5, s + 0.5, -2.0 * s])

    else:
        raise ValueError("Unsupported side with " + str(n_nodes) + " nodes")

    return N, dN
