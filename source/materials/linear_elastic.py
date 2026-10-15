import numpy


PLANE_STRESS = "plane_stress"
PLANE_STRAIN = "plane_strain"


class LinearElastic:
    """Material elástico lineal isótropo.

    E          módulo de elasticidad [Pa]
    nu         coeficiente de Poisson [-]
    density    densidad [kg/m3], se usa para el peso propio
    hypothesis PLANE_STRESS (szz = 0, cuerpo delgado) o
               PLANE_STRAIN (ezz = 0, cuerpo largo en z)
    """

    def __init__(self, E, nu, density=0.0, hypothesis=PLANE_STRESS):
        if E <= 0.0:
            raise ValueError("Expected positive E")

        if nu < 0.0 or nu >= 0.5:
            raise ValueError("Expected 0 <= nu < 0.5")

        if density < 0.0:
            raise ValueError("Expected non-negative density")

        self.E = E
        self.nu = nu
        self.density = density
        self.hypothesis = hypothesis

        if hypothesis == PLANE_STRESS:
            self.D = self.plane_stress_matrix()
        elif hypothesis == PLANE_STRAIN:
            self.D = self.plane_strain_matrix()
        else:
            raise ValueError("Unknown hypothesis: " + str(hypothesis))

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

    def plane_strain_matrix(self):
        """Matriz constitutiva D de estado plano de deformación (ezz = 0)."""
        E = self.E
        nu = self.nu

        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))

        return c * numpy.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ])
