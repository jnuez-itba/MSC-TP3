import numpy

from source.mesh.mesh import Mesh
from source.materials.linear_elastic import LinearElastic
from source.elements.bar2d import Bar2D
from source.systems.dense_matrix import DenseMatrix
from source.systems.boundary_conditions import apply_dirichlet_bc
from source.solvers.linear import solve
from source.output.paraview import write_results


mesh = Mesh()
mesh.read("meshes/roof.mesh")
mesh.dofs_per_node = Bar2D.dofs_per_node

material = LinearElastic(200.0e9, 0.0)
area = 2.0e-3


elements = []

for e in range(mesh.number_of_elements()):
    if mesh.element_types[e] != "BAR2D":
        raise ValueError("Expected BAR2D")

    element = Bar2D(
        mesh.element_global_ids[e],
        mesh.connectivity[e],
        mesh,
        material,
        area,
    )

    elements.append(element)

number_of_dofs = mesh.number_of_dofs()
K = DenseMatrix(number_of_dofs)

for element in elements:
    K.scatter_local_to_global(element)

f = numpy.zeros(number_of_dofs)

left_support_dofs = mesh.node_global_dofs(0)
right_support_dofs = mesh.node_global_dofs(12)
load_node_dofs = mesh.node_global_dofs(18)

f[load_node_dofs[1]] = -10000.0


apply_dirichlet_bc(K, f, left_support_dofs[0], 0.0)
apply_dirichlet_bc(K, f, left_support_dofs[1], 0.0)
apply_dirichlet_bc(K, f, right_support_dofs[1], 0.0)

u = solve(K, f)

write_results("results/bar_roof.vtu", mesh, elements, u)
