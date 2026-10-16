"""Gráfico de la configuración deformada con matplotlib.

    plot_deformed(mesh, elements, u, field="sigma_1", scale=100)

dibuja la malla deformada (desplazamientos multiplicados por 'scale'),
coloreada con el campo elegido, y la malla original en gris claro de fondo.
"""

import math
import os

import numpy
import matplotlib.pyplot as pyplot
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from source.elements.element2d import side_shape_functions
from source.output.postprocess import nodal_displacements
from source.output.postprocess import stress_fields


# Cada elemento se divide en triángulos usando sus propios nodos, para que
# matplotlib pueda interpolar los colores dentro del elemento.
SUBTRIANGLES = {
    "TRI3": [(0, 1, 2)],
    "QUAD4": [(0, 1, 2), (0, 2, 3)],
    "TRI6": [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
    "QUAD9": [
        (0, 4, 8), (0, 8, 7), (4, 1, 5), (4, 5, 8),
        (7, 8, 6), (7, 6, 3), (8, 5, 2), (8, 2, 6),
    ],
}

# Campos disponibles: nombre -> (etiqueta, factor de unidades, unidad)
FIELDS = {
    "sigma_1": ("Tensión principal máxima $\\sigma_1$", 1.0e-6, "MPa"),
    "sigma_2": ("Tensión principal mínima $\\sigma_2$", 1.0e-6, "MPa"),
    "sigma_xx": ("$\\sigma_{xx}$", 1.0e-6, "MPa"),
    "sigma_yy": ("$\\sigma_{yy}$", 1.0e-6, "MPa"),
    "tau_xy": ("$\\tau_{xy}$", 1.0e-6, "MPa"),
    "sigma_zz": ("$\\sigma_{zz}$ (fuera del plano)", 1.0e-6, "MPa"),
    "von_mises": ("Tensión de von Mises", 1.0e-6, "MPa"),
    "displacement": ("Desplazamiento total $|u|$", 1.0e3, "mm"),
}


def nice_number(value):
    """Redondea a 1, 2 o 5 x 10^k (para que el factor de escala sea legible)."""
    if value <= 0.0:
        return 1.0

    exponent = math.floor(math.log10(value))
    fraction = value / 10.0 ** exponent

    if fraction < 1.5:
        nice = 1.0
    elif fraction < 3.5:
        nice = 2.0
    elif fraction < 7.5:
        nice = 5.0
    else:
        nice = 10.0

    return nice * 10.0 ** exponent


def automatic_scale(mesh, u, fraction=0.05):
    """Factor de escala para que el desplazamiento máximo dibujado sea
    'fraction' del tamaño del modelo."""
    xs = [node.x for node in mesh.nodes]
    ys = [node.y for node in mesh.nodes]
    size = max(max(xs) - min(xs), max(ys) - min(ys))

    displacements = nodal_displacements(mesh, u)
    max_displacement = numpy.sqrt((displacements ** 2).sum(axis=1)).max()

    if max_displacement == 0.0:
        return 1.0

    return nice_number(fraction * size / max_displacement)


def field_values(mesh, elements, u, field):
    if field not in FIELDS:
        raise ValueError(
            "Unknown field '" + field + "'. Options: " + ", ".join(FIELDS.keys())
        )

    if field == "displacement":
        displacements = nodal_displacements(mesh, u)
        return numpy.sqrt((displacements ** 2).sum(axis=1))

    return stress_fields(mesh, elements, u)[field]


def side_curve(coordinates, n_points=9):
    """Puntos a lo largo de un lado (recto o curvo) para dibujarlo."""
    points = []

    for k in range(n_points):
        s = -1.0 + 2.0 * k / (n_points - 1)
        N, dN = side_shape_functions(len(coordinates), s)
        points.append(N @ coordinates)

    return numpy.array(points)


def draw_edges(axes, elements, coordinates, color, linewidth, linestyle="-"):
    for element in elements:
        for side in element.side_nodes:
            node_ids = [element.node_global_ids[a] for a in side]
            curve = side_curve(coordinates[node_ids])
            axes.plot(curve[:, 0], curve[:, 1], color=color,
                      linewidth=linewidth, linestyle=linestyle)


def plot_deformed(mesh, elements, u, field="sigma_1", scale=None,
                  filename=None, show=True, title=None, color_limit=None):
    """Dibuja la configuración deformada coloreada con un campo.

    field     uno de FIELDS: sigma_1, sigma_2, sigma_xx, sigma_yy, tau_xy,
              sigma_zz, von_mises, displacement
    scale     factor de amplificación de los desplazamientos. Si es None se
              elige uno automático (desplazamiento máximo = 5% del modelo).
    filename  si se indica, guarda la figura (por ejemplo results/fig.png)
    show      abre la ventana con el gráfico
    color_limit  límite de la escala de colores, en las unidades del gráfico
              (MPa o mm). Los valores que lo superan se pintan con el color
              extremo (escala saturada). Sirve cuando hay picos singulares,
              por ejemplo en las esquinas de un empotramiento.

    Devuelve el factor de escala usado.
    """
    if scale is None:
        scale = automatic_scale(mesh, u)

    label, unit_factor, unit = FIELDS[field]
    values = field_values(mesh, elements, u, field) * unit_factor

    original = numpy.array([[node.x, node.y] for node in mesh.nodes])
    deformed = original + scale * nodal_displacements(mesh, u)

    triangles = []
    for e in range(len(elements)):
        connectivity = mesh.connectivity[e]
        for a, b, c in SUBTRIANGLES[mesh.element_types[e]]:
            triangles.append([connectivity[a], connectivity[b], connectivity[c]])

    triangulation = tri.Triangulation(deformed[:, 0], deformed[:, 1], triangles)

    # Color: campos con signo -> divergente simétrico centrado en cero (rojo
    # tracción, azul compresión, gris en cero). Campos de un solo signo ->
    # secuencial de un tono.
    v_min = values.min()
    v_max = values.max()
    extend = "neither"

    if v_min < 0.0 < v_max:
        colormap = "RdBu_r"
        limit = max(-v_min, v_max)
        # solo se satura si el pico es claramente mayor (más de 10 %)
        if color_limit is not None and color_limit < 0.9 * limit:
            limit = color_limit
            extend = "both"
        norm = Normalize(vmin=-limit, vmax=limit)
    elif v_max <= 0.0:
        colormap = "Blues_r"
        norm = None
        if color_limit is not None and -color_limit > 0.9 * v_min:
            norm = Normalize(vmin=-color_limit, vmax=v_max)
            extend = "min"
    else:
        colormap = "Reds"
        norm = None
        if color_limit is not None and color_limit < 0.9 * v_max:
            norm = Normalize(vmin=v_min, vmax=color_limit)
            extend = "max"

    # Tamaño de figura según la forma del modelo: si es muy alargado (una
    # viga), la barra de colores va abajo; si no, a la derecha.
    width = deformed[:, 0].max() - deformed[:, 0].min()
    height = deformed[:, 1].max() - deformed[:, 1].min()
    elongated = width > 2.0 * height

    if elongated:
        figure, axes = pyplot.subplots(figsize=(10, 1.0 + 8.0 * height / width + 1.5))
    else:
        figure, axes = pyplot.subplots(figsize=(2.0 + 7.0 * width / height + 1.5, 9))

    draw_edges(axes, elements, original, color="#c8c8c8", linewidth=0.6, linestyle="--")

    image = axes.tripcolor(triangulation, values, shading="gouraud",
                           cmap=colormap, norm=norm)

    draw_edges(axes, elements, deformed, color="#404040", linewidth=0.4)

    divider = make_axes_locatable(axes)

    if elongated:
        colorbar_axes = divider.append_axes("bottom", size=0.15, pad=0.6)
        colorbar = figure.colorbar(image, cax=colorbar_axes, orientation="horizontal",
                                   extend=extend)
    else:
        colorbar_axes = divider.append_axes("right", size="4%", pad=0.15)
        colorbar = figure.colorbar(image, cax=colorbar_axes, extend=extend)

    colorbar.set_label(label + " [" + unit + "]")

    if title is None:
        title = label
    subtitle = "Deformada amplificada x" + format(scale, "g")
    if extend != "neither":
        subtitle = (subtitle + "\nescala saturada; valores reales de "
                    + format(v_min, ".3g") + " a " + format(v_max, ".3g") + " " + unit)
    axes.set_title(title + "\n" + subtitle)

    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_aspect("equal")
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

    return scale


# Colores fijos para marcar conjuntos (siempre en el mismo orden)
SET_COLORS = ["#2a6fdb", "#e07b00", "#1b9e77", "#c2185b", "#7b4fd6", "#8c6d1f"]


def plot_mesh(mesh, elements, side_sets=None, node_sets=None, filename=None,
              show=True, title="Malla"):
    """Dibuja la malla y resalta conjuntos de lados y de nodos.

    side_sets  lista de nombres de SideSets a resaltar (líneas gruesas)
    node_sets  lista de nombres de NodeSets a resaltar (puntos)
    """
    if side_sets is None:
        side_sets = []
    if node_sets is None:
        node_sets = []

    coordinates = numpy.array([[node.x, node.y] for node in mesh.nodes])

    width = coordinates[:, 0].max() - coordinates[:, 0].min()
    height = coordinates[:, 1].max() - coordinates[:, 1].min()

    if width > 2.0 * height:
        figure, axes = pyplot.subplots(figsize=(10, 1.0 + 8.0 * height / width + 1.5))
    else:
        figure, axes = pyplot.subplots(figsize=(2.0 + 7.0 * width / height + 1.5, 9))

    draw_edges(axes, elements, coordinates, color="#9a9a9a", linewidth=0.5)

    axes.plot(coordinates[:, 0], coordinates[:, 1], linestyle="none",
              marker=".", markersize=2, color="#606060")

    color_index = 0

    for name in side_sets:
        color = SET_COLORS[color_index % len(SET_COLORS)]
        color_index = color_index + 1

        first = True
        for element_id, side in mesh.side_sets[name]:
            element = elements[element_id]
            node_ids = [element.node_global_ids[a] for a in element.side_nodes[side]]
            curve = side_curve(coordinates[node_ids])
            axes.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3.0,
                      label="SideSet " + name if first else None)
            first = False

    for name in node_sets:
        color = SET_COLORS[color_index % len(SET_COLORS)]
        color_index = color_index + 1

        ids = mesh.node_sets[name]
        axes.plot(coordinates[ids, 0], coordinates[ids, 1], linestyle="none",
                  marker="o", markersize=8, color=color, markeredgecolor="white",
                  label="NodeSet " + name)

    n_nodes = mesh.number_of_nodes()
    n_elements = mesh.number_of_elements()
    axes.set_title(title + "\n" + str(n_elements) + " elementos "
                   + mesh.element_types[0] + ", " + str(n_nodes) + " nodos")

    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_aspect("equal")

    if len(side_sets) + len(node_sets) > 0:
        axes.legend(loc="upper right", fontsize=8)

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
