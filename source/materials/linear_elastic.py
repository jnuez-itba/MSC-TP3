import numpy


class LinearElastic:
    """Material elástico lineal isótropo.

    E       módulo de elasticidad [Pa]
    nu      coeficiente de Poisson [-]
    density densidad [kg/m3], se usa para el peso propio
    """

    def __init__(self, E, nu, density=0.0):
        if E <= 0.0:
            raise ValueError("Expected positive E")

        if nu < 0.0 or nu >= 0.5:
            raise ValueError("Expected 0 <= nu < 0.5")

        self.E = E
        self.nu = nu
        self.density = density

        self.D = self.plane_stress_matrix()

    def plane_stress_matrix(self):
        """Matriz constitutiva D de estado plano de tensión.

        Relaciona sigma = D * epsilon con la notación de Voigt
        sigma   = [sxx, syy, txy]
        epsilon = [exx, eyy, gxy]   (gxy = 2 * exy, deformación ingenieril)
        """
        E = self.E
        nu = self.nu

        c = E / (1.0 - nu * nu)

        return c * numpy.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
