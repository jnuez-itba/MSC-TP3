# TP3 - Estado plano de tensión

30.53 Mecánica del Sólido Computacional - ITBA

Código de elementos finitos en Python para modelar el equilibrio estático de un
sólido elástico lineal en estado plano de tensión. Parte de la estructura vista
en clase para barras 2D (`bar.py`, `bar_roof.py`) y la extiende a elementos de
sólido 2D.

## Elementos disponibles

| Familia | Orden | Nodos | Nombre | Integración |
|---------|-------|-------|--------|-------------|
| TRI     | 1     | 3     | Tri3   | 1 punto     |
| TRI     | 2     | 6     | Tri6   | 3 puntos    |
| QUAD    | 1     | 4     | Quad4  | 2 x 2       |
| QUAD    | 2     | 9     | Q9     | 3 x 3       |

Todos pasan el patch test (`python patch_test.py`).

## Estructura

```
source/
  elements/      elementos finitos (Element2D, Tri3, Quad4, Tri6, Quad9)
  materials/     modelos de material
  mesh/          nodos y lectura de mallas
  quadrature/    reglas de integración de Gauss
  systems/       matriz global y condiciones de borde
  solvers/       resolución del sistema lineal
  output/        exportación de resultados (ParaView)
  verification/  patch test (por desplazamientos y por fuerzas)
meshes/          archivos de malla
tests/           tests automáticos (pytest)
```

## Requisitos

```
pip install -r requirements.txt
```

## Uso

```
python patch_test.py     # patch test de los 4 elementos (versiones A y B)
pytest                   # todos los tests
python bar.py            # ejemplos de barras de la clase
python bar_roof.py
```

## Cargas, apoyos y reacciones

Flujo típico de un análisis (ver `source/systems/`):

```python
mesh = Mesh()
mesh.read("meshes/modelo.mesh")
mesh.dofs_per_node = 2
elements = create_elements(mesh)

K = DenseMatrix(mesh.number_of_dofs())
for element in elements:
    K.scatter_local_to_global(element)

f = numpy.zeros(mesh.number_of_dofs())
add_self_weight(f, elements)                          # b = (0, -rho g)
add_side_pressure(f, mesh, elements, "aguas_arriba",  # p = rho_w g (nivel - y)
                  hydrostatic_pressure(28.0))

K_original = K.values.copy()                          # para las reacciones
f_original = f.copy()

fix_node_set(K, f, mesh, "base", (0, 1))              # u = v = 0

u = solve(K, f)
R = compute_reactions(K_original, f_original, u)      # R = K u - f
Fx, Fy, Mz = node_set_reaction_resultant(mesh, R, "base", (18.0, 0.0))
```

- `add_self_weight`: peso propio con la densidad de cada material.
- `add_side_traction` / `add_side_pressure`: tracción o presión sobre un
  SideSet. La presión es positiva en compresión.
- `fix_node_set`: componentes `(0, 1)` empotramiento, `(1,)` solo v, `(0,)`
  solo u.
- `node_set_reaction_resultant`: fuerza y momento de las reacciones respecto
  de un punto (positivo antihorario).

Las cargas sobre lados se integran con 3 puntos de Gauss: es exacto para una
presión lineal. Si la presión tiene un quiebre (nivel del agua), conviene que
la malla tenga un nodo en ese nivel.

## Formato de malla

Archivo de texto plano dividido en secciones `$Nombre` ... `$EndNombre`.
Todo lo que sigue a `#` es comentario. Nodos y elementos se numeran desde 1,
consecutivos y en orden. En `meshes/ejemplo_formato.mesh` hay un ejemplo
completo comentado.

### `$Parameters` (opcional)

```
$Parameters
thickness 1.0
$EndParameters
```

Espesor del sólido (por defecto 1).

### `$Materials`

```
$Materials
# id  tipo                  E        nu    densidad
1     ELASTIC_PLANE_STRESS  21.0e9   0.2   2400.0
$EndMaterials
```

| Tipo                   | Hipótesis                              |
|------------------------|----------------------------------------|
| `ELASTIC_PLANE_STRESS` | tensión plana (szz = 0), cuerpo delgado |
| `ELASTIC_PLANE_STRAIN` | deformación plana (ezz = 0), cuerpo largo |

### `$Nodes`

```
$Nodes
4            # cantidad
1 0.0 0.0    # id x y
...
$EndNodes
```

### `$Elements`

```
$Elements
2                                   # cantidad
# id  familia  orden  material  nodos
1     QUAD     1      1         1 2 3 4
2     TRI      2      1         2 5 3 6 7 8
$EndElements
```

- **familia**: `TRI` o `QUAD`.
- **orden** de interpolación: `1` (lineal) o `2` (cuadrático).
- **material**: id de un material de `$Materials`.
- **nodos**: vértices en sentido antihorario, después los nodos de lado y
  (en el Q9) el central:
  - TRI 2: vértices 1 2 3, lados 1-2, 2-3, 3-1.
  - QUAD 2: vértices 1 2 3 4, lados 1-2, 2-3, 3-4, 4-1, centro.

Elementos vecinos deben tener el mismo orden para que la malla sea conforme.

### `$NodeSets` (opcional)

Grupos de nodos con nombre, para aplicar apoyos. `nombre cantidad` y los ids
(pueden ocupar varias líneas).

```
$NodeSets
base 3
1 2 3
$EndNodeSets
```

### `$SideSets` (opcional)

Grupos de lados de elementos con nombre, para aplicar cargas distribuidas.
`nombre cantidad` y pares `elemento lado`. El lado k va del vértice k al
vértice k+1 (el último vuelve al vértice 1).

```
$SideSets
aguas_arriba 2
4 4
8 4
$EndSideSets
```

### Formato de la cátedra

Los archivos sin secciones (como `meshes/roof.mesh`) se siguen leyendo con el
formato original de barras.
