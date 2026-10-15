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
  verification/  patch test
meshes/          archivos de malla
tests/           tests automáticos (pytest)
```

## Requisitos

```
pip install -r requirements.txt
```

## Uso

```
python patch_test.py     # patch test de los 4 elementos
pytest                   # todos los tests
python bar.py            # ejemplos de barras de la clase
python bar_roof.py
```

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
