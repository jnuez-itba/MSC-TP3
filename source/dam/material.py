"""Material de la presa: hormigón H21.

La cátedra indicó resolver en ESTADO PLANO DE DEFORMACIÓN (ezz = 0): la
presa es muy larga en la dirección perpendicular a la sección, así que cada
rebanada no puede deformarse en z. Todos los scripts de la presa toman el
material de acá.
"""

from source.materials.linear_elastic import LinearElastic
from source.materials.linear_elastic import PLANE_STRAIN


E_CONCRETE = 21.0e9          # módulo de elasticidad [Pa]
NU_CONCRETE = 0.2            # coeficiente de Poisson
DENSITY_CONCRETE = 2400.0    # densidad [kg/m3]


def dam_material(hypothesis=PLANE_STRAIN, density=DENSITY_CONCRETE):
    """Hormigón de la presa. Por defecto en deformación plana."""
    return LinearElastic(E_CONCRETE, NU_CONCRETE, density, hypothesis)
