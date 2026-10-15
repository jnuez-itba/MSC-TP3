"""Patch test.

Versión A, por desplazamientos impuestos (run_patch_test):

1. Se arma un parche de elementos distorsionados con nodos interiores.
2. En los nodos del borde (NodeSet "boundary" del archivo de malla) se impone
   un campo lineal exacto:
       u = a0 + a1 x + a2 y
       v = b0 + b1 x + b2 y
3. Los nodos interiores quedan libres y sin carga. Se resuelve K u = f.
4. El elemento pasa si:
     - los nodos interiores reproducen el campo lineal exacto, y
     - la tensión en todos los puntos de Gauss es la constante D * epsilon.

Versión B, por fuerzas (run_patch_test_forces):

1. Mismo parche, pero solo con los apoyos mínimos para evitar el movimiento de
   cuerpo rígido (NodeSets "pin" y "roller").
2. Sobre el contorno (SideSet "boundary_sides") se aplica la tracción que
   corresponde al estado de tensión constante: t = sigma . n
3. Se resuelve y se verifica que la tensión en todos los puntos de Gauss sea
   la constante. Prueba además la integración de cargas sobre los lados.

El material, el espesor y el tipo de elemento se leen del archivo de malla.
"""

import numpy

from source.mesh.mesh import Mesh
from source.elements.factory import create_elements
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.systems.boundary_conditions import fix_node_set
from source.systems.loads import add_side_traction
from source.solvers.linear import solve


# Campo lineal impuesto (coeficientes arbitrarios)
A = [1.0e-3, 2.0e-3, -1.0e-3]
B = [-2.0e-3, 0.5e-3, -3.0e-3]


def exact_displacement(x, y):
    u = A[0] + A[1] * x + A[2] * y
    v = B[0] + B[1] * x + B[2] * y

    return u, v


def exact_strain():
    # exx = du/dx, eyy = dv/dy, gxy = du/dy + dv/dx
    return numpy.array([A[1], B[2], A[2] + B[1]])


def run_patch_test(mesh_filename):
    """Corre el patch test y devuelve un diccionario con los errores."""
    mesh = Mesh()
    mesh.read(mesh_filename)
    mesh.dofs_per_node = 2

    if "boundary" not in mesh.node_sets:
        raise ValueError("Patch mesh needs a NodeSet called 'boundary'")

    elements = create_elements(mesh)

    # Ensamblado (código de la cátedra, sin cambios)
    number_of_dofs = mesh.number_of_dofs()
    K = DenseMatrix(number_of_dofs)

    for element in elements:
        K.scatter_local_to_global(element)

    f = numpy.zeros(number_of_dofs)

    # Desplazamientos exactos impuestos en el borde
    fixed = mesh.node_sets["boundary"]

    for node_id in fixed:
        node = mesh.node(node_id)
        dofs = mesh.node_global_dofs(node_id)
        u_exact, v_exact = exact_displacement(node.x, node.y)

        apply_dirichlet_bc(K, f, dofs[0], u_exact)
        apply_dirichlet_bc(K, f, dofs[1], v_exact)

    u = solve(K, f)

    # Error en desplazamientos de los nodos interiores
    displacement_error = 0.0
    displacement_scale = 0.0
    interior = []

    for node in mesh.nodes:
        dofs = mesh.node_global_dofs(node.global_id)
        u_exact, v_exact = exact_displacement(node.x, node.y)

        displacement_scale = max(displacement_scale, abs(u_exact), abs(v_exact))

        if node.global_id not in fixed:
            interior.append(node.global_id)
            displacement_error = max(
                displacement_error,
                abs(u[dofs[0]] - u_exact),
                abs(u[dofs[1]] - v_exact),
            )

    # Error en tensiones en todos los puntos de Gauss
    stress_error = 0.0
    stress_scale = 0.0

    for element in elements:
        stress_exact = element.material.D @ exact_strain()
        stress_scale = max(stress_scale, numpy.abs(stress_exact).max())

        for stress in element.stress_at_gauss_points(u):
            stress_error = max(stress_error, numpy.abs(stress - stress_exact).max())

    return {
        "elements": len(elements),
        "interior_nodes": len(interior),
        "displacement_error": displacement_error / displacement_scale,
        "stress_error": stress_error / stress_scale,
        "stress_exact": elements[0].material.D @ exact_strain(),
    }


def run_patch_test_forces(mesh_filename):
    """Patch test por fuerzas. Devuelve un diccionario con el error en tensiones."""
    mesh = Mesh()
    mesh.read(mesh_filename)
    mesh.dofs_per_node = 2

    for name in ["pin", "roller"]:
        if name not in mesh.node_sets:
            raise ValueError("Patch mesh needs a NodeSet called '" + name + "'")

    if "boundary_sides" not in mesh.side_sets:
        raise ValueError("Patch mesh needs a SideSet called 'boundary_sides'")

    elements = create_elements(mesh)

    # Tensión constante exacta (un solo material en el parche)
    D = elements[0].material.D
    sxx, syy, txy = D @ exact_strain()

    def traction(x, y, normal):
        # t = sigma . n
        tx = sxx * normal[0] + txy * normal[1]
        ty = txy * normal[0] + syy * normal[1]
        return tx, ty

    number_of_dofs = mesh.number_of_dofs()
    K = DenseMatrix(number_of_dofs)

    for element in elements:
        K.scatter_local_to_global(element)

    f = numpy.zeros(number_of_dofs)
    add_side_traction(f, mesh, elements, "boundary_sides", traction)

    # Apoyos mínimos: un nodo fijo en x e y, otro fijo en y
    fix_node_set(K, f, mesh, "pin", (0, 1))
    fix_node_set(K, f, mesh, "roller", (1,))

    u = solve(K, f)

    stress_exact = numpy.array([sxx, syy, txy])
    stress_error = 0.0

    for element in elements:
        for stress in element.stress_at_gauss_points(u):
            stress_error = max(stress_error, numpy.abs(stress - stress_exact).max())

    return {
        "elements": len(elements),
        "stress_error": stress_error / numpy.abs(stress_exact).max(),
    }
