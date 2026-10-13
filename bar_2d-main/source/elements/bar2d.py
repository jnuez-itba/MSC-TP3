import math

import numpy


class Bar2D:
    dofs_per_node = 2

    def __init__(self, global_id, node_global_ids, mesh, material, area):
        self.global_id = global_id
        self.node_global_ids = node_global_ids
        self.mesh = mesh
        self.material = material
        self.area = area

        node_1 = mesh.node(node_global_ids[0])
        node_2 = mesh.node(node_global_ids[1])

        dx = node_2.x - node_1.x
        dy = node_2.y - node_1.y

        self.length = math.sqrt(dx * dx + dy * dy)

        if self.length <= 0.0:
            raise ValueError("Expected nonzero length")

        c = dx / self.length
        s = dy / self.length

        k = material.E * area / self.length

        T = numpy.array([
            [c, s, 0.0, 0.0],
            [0.0, 0.0, c, s],
        ])

        Ke_local = numpy.array([
            [k, -k],
            [-k, k],
        ])

        self.T = T
        self.Ke = T.transpose() @ Ke_local @ T

    def global_dofs(self):
        dofs = []

        for node_global_id in self.node_global_ids:
            for dof in self.mesh.node_global_dofs(node_global_id):
                dofs.append(dof)

        return dofs

    def stiffness_matrix(self):
        return self.Ke

    def axial_strain(self, u):
        dofs = self.global_dofs()

        ue = numpy.zeros(len(dofs))

        for i in range(len(dofs)):
            ue[i] = u[dofs[i]]

        u_axial = self.T @ ue

        return (u_axial[1] - u_axial[0]) / self.length

    def axial_stress(self, u):
        return self.material.E * self.axial_strain(u)
