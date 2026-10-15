"""Cálculo manual de la estabilidad de la presa (cuerpo rígido).

Momentos respecto del PIE (esquina aguas abajo de la base), por metro de
presa (espesor t):

  Estabilizantes (tienden a apoyar la presa):
    - peso del hormigón  W = rho_c g A t, aplicado en el centroide
    - si el paramento aguas arriba está inclinado, peso del agua que queda
      encima de él: Fv = rho_w g (a hw^2 / 2H) t, en el centroide de esa cuña

  Desestabilizante (tiende a volcarla girando alrededor del pie):
    - empuje horizontal del agua  Fh = rho_w g hw^2 / 2 t, a hw/3 de la base

  Suelo impermeable -> no hay subpresión en la base.

  Factor de seguridad al vuelco:  FS = M_estabilizante / M_desestabilizante

Distribución de tensiones en la base (ley trapecial, teoría de vigas):
  la resultante vertical N = W + Fv corta la base a una distancia
  x_R = (M_est - M_des) / N del pie. Con excentricidad e = B/2 - x_R:

      q_pie   = N/B (1 + 6 e / B)      (compresión, positiva)
      q_talón = N/B (1 - 6 e / B)

  Si la resultante cae dentro del tercio central (B/3 <= x_R <= 2B/3), toda
  la base está comprimida.
"""

from source.systems.loads import GRAVITY


def stability(geometry, gravity=GRAVITY, thickness=1.0):
    """Devuelve un diccionario con fuerzas, momentos y factores [N, m]."""
    H = geometry.height
    B = geometry.base
    a = geometry.upstream_batter
    hw = geometry.water_level()
    t = thickness

    x_toe, y_toe = geometry.toe()

    # Peso del hormigón
    A = geometry.area()
    x_c, y_c = geometry.centroid()
    W = geometry.concrete_density * gravity * A * t
    M_weight = W * (x_toe - x_c)

    # Empuje horizontal del agua
    Fh = geometry.water_density * gravity * hw ** 2 / 2.0 * t
    M_horizontal = Fh * hw / 3.0

    # Peso del agua sobre el paramento inclinado (cuña triangular con
    # vértices (-a, 0), (-a, hw) y (x_face(hw), hw))
    if a > 0.0:
        x_face = -a + a * hw / H
        wedge_area = (x_face + a) * hw / 2.0
        x_wedge = (-a - a + x_face) / 3.0
        Fv = geometry.water_density * gravity * wedge_area * t
        M_vertical = Fv * (x_toe - x_wedge)
    else:
        Fv = 0.0
        M_vertical = 0.0

    M_stabilizing = M_weight + M_vertical
    M_overturning = M_horizontal

    N = W + Fv
    x_R = (M_stabilizing - M_overturning) / N      # desde el pie
    e = B / 2.0 - x_R

    q_toe = N / (B * t) * (1.0 + 6.0 * e / B)
    q_heel = N / (B * t) * (1.0 - 6.0 * e / B)

    return {
        "area": A,
        "centroid": (x_c, y_c),
        "weight": W,
        "water_horizontal": Fh,
        "water_vertical": Fv,
        "moment_weight": M_weight,
        "moment_water_vertical": M_vertical,
        "moment_stabilizing": M_stabilizing,
        "moment_overturning": M_overturning,
        "safety_factor": M_stabilizing / M_overturning,
        "normal_force": N,
        "resultant_from_toe": x_R,
        "eccentricity": e,
        "middle_third": B / 3.0 <= x_R <= 2.0 * B / 3.0,
        "base_pressure_toe": q_toe,
        "base_pressure_heel": q_heel,
        "required_friction": Fh / N,
    }


def trapezoidal_base_pressure(geometry, result, x):
    """Presión de contacto (compresión positiva) en la base según la ley
    trapecial, en la abscisa x."""
    x_heel = geometry.heel()[0]
    x_toe = geometry.toe()[0]

    s = (x - x_heel) / (x_toe - x_heel)

    return (1.0 - s) * result["base_pressure_heel"] + s * result["base_pressure_toe"]
