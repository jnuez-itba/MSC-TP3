from source.mesh.node import Node
from source.materials.factory import create_material


# Cantidad de nodos según familia y orden de interpolación
NODES_PER_ELEMENT = {
    ("TRI", 1): 3,
    ("TRI", 2): 6,
    ("QUAD", 1): 4,
    ("QUAD", 2): 9,
}

# Nombre del tipo de elemento según familia y orden
ELEMENT_NAMES = {
    ("TRI", 1): "TRI3",
    ("TRI", 2): "TRI6",
    ("QUAD", 1): "QUAD4",
    ("QUAD", 2): "QUAD9",
}

# Cantidad de lados de cada familia (para los SideSets)
SIDES_PER_FAMILY = {
    "TRI": 3,
    "QUAD": 4,
}


class Mesh:
    """Malla de elementos finitos.

    Lee dos formatos:
      - el formato original de la cátedra (barras), sin secciones;
      - el formato por secciones ($Parameters, $Materials, $Nodes, $Elements,
        $NodeSets, $SideSets) para sólidos 2D. Ver README.md.

    Toda la numeración interna empieza en 0. En los archivos empieza en 1.
    """

    def __init__(self):
        self.dim = 0
        self.dofs_per_node = 0
        self.nodes = []
        self.element_global_ids = []
        self.element_types = []
        self.connectivity = []

        # Solo para el formato por secciones
        self.thickness = 1.0
        self.materials = {}
        self.element_families = []
        self.element_orders = []
        self.element_material_ids = []
        self.node_sets = {}
        self.side_sets = {}

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def read(self, filename):
        file = open(filename, "r")

        lines = []

        for raw_line in file:
            line = raw_line.split("#")[0]
            line = line.strip()

            if line != "":
                lines.append(line)

        file.close()

        if len(lines) == 0:
            raise ValueError("Empty mesh file: " + filename)

        if lines[0].startswith("$"):
            self.read_sections(lines)
        else:
            self.read_legacy(lines)

    def read_legacy(self, lines):
        """Formato original de la cátedra (sin cambios)."""
        header = lines[0].split()

        n_nodes = int(header[0])
        n_elements = int(header[1])
        dim = int(header[2])

        self.dim = dim

        line_number = 1

        for i in range(n_nodes):
            values = lines[line_number].split()
            line_number = line_number + 1

            global_id = int(values[0]) - 1

            if global_id != i:
                raise ValueError("Unexpected node numbering")

            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

        for i in range(n_elements):
            values = lines[line_number].split()
            line_number = line_number + 1

            element_global_id = int(values[0]) - 1
            element_type = values[1]

            element_conn = []

            for j in range(2, len(values)):
                element_conn.append(int(values[j]) - 1)

            self.element_global_ids.append(element_global_id)
            self.element_types.append(element_type)
            self.connectivity.append(element_conn)

    def read_sections(self, lines):
        """Formato por secciones $Nombre ... $EndNombre."""
        sections = split_sections(lines)

        for required in ["Materials", "Nodes", "Elements"]:
            if required not in sections:
                raise ValueError("Missing section $" + required)

        self.dim = 2

        if "Parameters" in sections:
            self.read_parameters(sections["Parameters"])

        self.read_materials(sections["Materials"])
        self.read_nodes(sections["Nodes"])
        self.read_elements(sections["Elements"])

        if "NodeSets" in sections:
            self.read_node_sets(sections["NodeSets"])

        if "SideSets" in sections:
            self.read_side_sets(sections["SideSets"])

    def read_parameters(self, lines):
        for line in lines:
            values = line.split()

            if len(values) != 2:
                raise ValueError("Expected 'name value' in $Parameters: " + line)

            name = values[0].lower()

            if name == "thickness":
                self.thickness = float(values[1])

                if self.thickness <= 0.0:
                    raise ValueError("Expected positive thickness")
            else:
                raise ValueError("Unknown parameter: " + values[0])

    def read_materials(self, lines):
        for line in lines:
            values = line.split()

            material_id = int(values[0])
            material_type = values[1]
            parameters = values[2:]

            if material_id in self.materials:
                raise ValueError("Repeated material id: " + str(material_id))

            self.materials[material_id] = create_material(material_type, parameters)

    def read_nodes(self, lines):
        n_nodes = int(lines[0])

        if len(lines) - 1 != n_nodes:
            raise ValueError(
                "$Nodes declares " + str(n_nodes) + " nodes but has "
                + str(len(lines) - 1)
            )

        for i in range(n_nodes):
            values = lines[i + 1].split()

            global_id = int(values[0]) - 1

            if global_id != i:
                raise ValueError("Nodes must be numbered 1, 2, 3, ... in order")

            x = float(values[1])
            y = float(values[2])

            self.nodes.append(Node(global_id, x, y))

    def read_elements(self, lines):
        n_elements = int(lines[0])

        if len(lines) - 1 != n_elements:
            raise ValueError(
                "$Elements declares " + str(n_elements) + " elements but has "
                + str(len(lines) - 1)
            )

        for i in range(n_elements):
            values = lines[i + 1].split()

            element_global_id = int(values[0]) - 1

            if element_global_id != i:
                raise ValueError("Elements must be numbered 1, 2, 3, ... in order")

            family = values[1].upper()
            order = int(values[2])
            material_id = int(values[3])

            key = (family, order)

            if key not in NODES_PER_ELEMENT:
                raise ValueError(
                    "Element " + str(i + 1) + ": unknown family/order "
                    + family + " " + str(order)
                )

            if material_id not in self.materials:
                raise ValueError(
                    "Element " + str(i + 1) + ": unknown material "
                    + str(material_id)
                )

            element_conn = []

            for j in range(4, len(values)):
                node_id = int(values[j]) - 1

                if node_id < 0 or node_id >= len(self.nodes):
                    raise ValueError(
                        "Element " + str(i + 1) + ": unknown node " + values[j]
                    )

                element_conn.append(node_id)

            if len(element_conn) != NODES_PER_ELEMENT[key]:
                raise ValueError(
                    "Element " + str(i + 1) + ": " + family + " order "
                    + str(order) + " needs " + str(NODES_PER_ELEMENT[key])
                    + " nodes, got " + str(len(element_conn))
                )

            self.element_global_ids.append(element_global_id)
            self.element_types.append(ELEMENT_NAMES[key])
            self.element_families.append(family)
            self.element_orders.append(order)
            self.element_material_ids.append(material_id)
            self.connectivity.append(element_conn)

    def read_node_sets(self, lines):
        """Cada conjunto: 'nombre cantidad' y después los ids de nodo
        (pueden ocupar varias líneas)."""
        tokens = tokenize(lines)
        position = 0

        while position < len(tokens):
            name = tokens[position]
            count = int(tokens[position + 1])
            position = position + 2

            node_ids = []

            for k in range(count):
                node_id = int(tokens[position + k]) - 1

                if node_id < 0 or node_id >= len(self.nodes):
                    raise ValueError("NodeSet " + name + ": unknown node " + str(node_id + 1))

                node_ids.append(node_id)

            position = position + count

            self.node_sets[name] = node_ids

    def read_side_sets(self, lines):
        """Cada conjunto: 'nombre cantidad' y después pares 'elemento lado'.

        El lado k de un elemento va del vértice k al vértice k + 1 (y el
        último lado vuelve al vértice 1), siguiendo la numeración local.
        """
        tokens = tokenize(lines)
        position = 0

        while position < len(tokens):
            name = tokens[position]
            count = int(tokens[position + 1])
            position = position + 2

            sides = []

            for k in range(count):
                element_id = int(tokens[position + 2 * k]) - 1
                side = int(tokens[position + 2 * k + 1]) - 1

                if element_id < 0 or element_id >= len(self.connectivity):
                    raise ValueError(
                        "SideSet " + name + ": unknown element " + str(element_id + 1)
                    )

                n_sides = SIDES_PER_FAMILY[self.element_families[element_id]]

                if side < 0 or side >= n_sides:
                    raise ValueError(
                        "SideSet " + name + ": element " + str(element_id + 1)
                        + " has no side " + str(side + 1)
                    )

                sides.append((element_id, side))

            position = position + 2 * count

            self.side_sets[name] = sides

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_elements(self):
        return len(self.connectivity)

    def node(self, node_global_id):
        return self.nodes[node_global_id]

    def number_of_dofs(self):
        return self.number_of_nodes() * self.dofs_per_node

    def node_global_dofs(self, node_global_id):
        dofs = []

        for i in range(self.dofs_per_node):
            dofs.append(node_global_id * self.dofs_per_node + i)

        return dofs

    def element_material(self, element_index):
        return self.materials[self.element_material_ids[element_index]]


def split_sections(lines):
    """Agrupa las líneas en un diccionario {nombre de sección: [líneas]}."""
    sections = {}
    current = None

    for line in lines:
        if line.startswith("$End"):
            if current is None or line[4:] != current:
                raise ValueError("Unexpected " + line)

            current = None

        elif line.startswith("$"):
            if current is not None:
                raise ValueError("Section $" + current + " is not closed")

            current = line[1:]

            if current in sections:
                raise ValueError("Repeated section $" + current)

            sections[current] = []

        else:
            if current is None:
                raise ValueError("Line outside of a section: " + line)

            sections[current].append(line)

    if current is not None:
        raise ValueError("Section $" + current + " is not closed")

    return sections


def tokenize(lines):
    tokens = []

    for line in lines:
        for token in line.split():
            tokens.append(token)

    return tokens
