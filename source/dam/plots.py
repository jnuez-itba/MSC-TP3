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
