"""Gráficos específicos de la presa."""

import os

import numpy
import matplotlib.pyplot as pyplot

from source.dam.stability import trapezoidal_base_pressure


# Colores fijos por caso (siempre el mismo color para el mismo caso)
CASE_COLORS = ["#2a6fdb", "#e07b00", "#1b9e77", "#c2185b"]


def plot_base_pressure(geometry, stability_result, curves, filename=None,
                       show=True, clip=None):
    """Presión de contacto en la base (compresión positiva, MPa) a lo largo de x.

    curves  lista de (etiqueta, x, presión en Pa) de los modelos FEM
    clip    si se indica, límite del eje vertical en MPa (los picos de las
            esquinas pueden ser mucho mayores)

    Se compara con la ley trapecial del cálculo de cuerpo rígido.
    """
    x_heel = geometry.heel()[0]
    x_toe = geometry.toe()[0]

    figure, axes = pyplot.subplots(figsize=(9, 5))

    x_line = numpy.array([x_heel, x_toe])
    q_line = [trapezoidal_base_pressure(geometry, stability_result, x) * 1.0e-6
              for x in x_line]
    axes.plot(x_line, q_line, color="#303030", linewidth=2.0, linestyle="--",
              label="Ley trapecial (cuerpo rígido)")

    for k in range(len(curves)):
        label, x, q = curves[k]
        axes.plot(x, numpy.array(q) * 1.0e-6, color=CASE_COLORS[k % len(CASE_COLORS)],
                  linewidth=2.0, marker="o", markersize=3, label=label)

    axes.axhline(0.0, color="#a0a0a0", linewidth=0.8)

    axes.set_xlabel("x [m]   (talón a la izquierda, pie a la derecha)")
    axes.set_ylabel("Presión de contacto [MPa]  (compresión +)")
    axes.set_title("Distribución de tensiones normales en la base")

    if clip is not None:
        axes.set_ylim(-clip, clip)

    axes.grid(True, color="#e6e6e6", linewidth=0.6)
    axes.legend(loc="best", fontsize=9)
    figure.tight_layout()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        figure.savefig(filename, dpi=150)

    if show:
        pyplot.show()
    else:
        pyplot.close(figure)


def plot_safety_factor_vs_area(results, families, filename=None, show=True,
                               original=None):
    """Factor de seguridad al vuelco en función del área de hormigón, una
    curva por familia de variantes.

    results   {familia: [(valor, resultado)]}
    families  {familia: etiqueta}
    original  resultado de la presa del enunciado (se marca aparte)
    """
    figure, axes = pyplot.subplots(figsize=(9, 5.5))

    k = 0
    for family in results:
        points = sorted(results[family], key=lambda item: item[1]["area"])
        areas = [r["area"] for v, r in points]
        factors = [r["safety_factor"] for v, r in points]

        color = CASE_COLORS[k % len(CASE_COLORS)]
        axes.plot(areas, factors, color=color, linewidth=2.0, marker="o",
                  markersize=7, label=families[family])

        # etiqueta directa al final de cada curva
        axes.annotate(families[family], (areas[-1], factors[-1]),
                      textcoords="offset points", xytext=(6, 0),
                      va="center", fontsize=8, color="#404040")
        k = k + 1

    if original is not None:
        axes.plot([original["area"]], [original["safety_factor"]], linestyle="none",
                  marker="*", markersize=16, color="#303030",
                  label="Presa del enunciado")

    axes.set_xlabel("Área de hormigón [m²]  (volumen por metro de presa)")
    axes.set_ylabel("Factor de seguridad al vuelco  Ms / Md")
    axes.set_title("Variaciones de geometría: factor de seguridad vs hormigón")
    axes.grid(True, color="#e6e6e6", linewidth=0.6)
    axes.legend(loc="upper left", fontsize=9)

    x_min, x_max = axes.get_xlim()
    axes.set_xlim(x_min, x_max + 0.18 * (x_max - x_min))

    figure.tight_layout()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        figure.savefig(filename, dpi=150)

    if show:
        pyplot.show()
    else:
        pyplot.close(figure)


def plot_sections(designs, filename=None, show=True):
    """Dibuja el contorno de varias secciones superpuestas.

    designs  lista de (etiqueta, geometría, factor de seguridad)
    """
    figure, axes = pyplot.subplots(figsize=(8, 9))

    water_drawn = False

    for k in range(len(designs)):
        label, geometry, fs = designs[k]
        corners = geometry.corners()
        xs = [p[0] for p in corners] + [corners[0][0]]
        ys = [p[1] for p in corners] + [corners[0][1]]

        if k == 0:
            axes.fill(xs, ys, color="#d9d9d9", zorder=1)
            axes.plot(xs, ys, color="#303030", linewidth=2.0, zorder=3,
                      label=label + "  (FS = {:.2f})".format(fs))
        else:
            color = CASE_COLORS[(k - 1) % len(CASE_COLORS)]
            axes.plot(xs, ys, color=color, linewidth=2.0, linestyle="--", zorder=3,
                      label=label + "  (FS = {:.2f})".format(fs))

        if not water_drawn:
            hw = geometry.water_level()
            x_left = min(min(p[0] for p in d[1].corners()) for d in designs) - 2.0
            axes.plot([x_left, 0.0], [hw, hw], color="#2a6fdb", linewidth=1.2)
            axes.annotate("nivel del agua ({:g} m)".format(hw), (x_left, hw),
                          textcoords="offset points", xytext=(2, 4), fontsize=8,
                          color="#2a6fdb")
            water_drawn = True

    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_aspect("equal")
    axes.set_title("Secciones con la misma área de hormigón")
    axes.grid(True, color="#e6e6e6", linewidth=0.6)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=1)
    figure.tight_layout()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        figure.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        pyplot.show()
    else:
        pyplot.close(figure)
