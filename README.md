# TP3 - Elasticidad plana

30.53 Mecánica del Sólido Computacional - ITBA

Código de elementos finitos en Python para modelar el equilibrio estático de un
sólido elástico lineal en estado plano. Parte de la estructura vista en clase
para barras 2D (`bar.py`, `bar_roof.py`) y la extiende a elementos de sólido 2D.

El material admite **estado plano de tensión** (szz = 0) y **estado plano de
deformación** (ezz = 0); la hipótesis se elige en el archivo de malla. La presa
se resuelve en **estado plano de deformación**, según indicó la cátedra
(`source/dam/material.py`): es muy larga en la dirección perpendicular a la
sección. En deformación plana aparece la tensión fuera del plano
szz = nu (sxx + syy), que también se informa.

## Inicio rápido

```
pip install -r requirements.txt
python ejecutar_todo.py
```

`ejecutar_todo.py` corre todo el trabajo en orden, explicando cada paso en la
terminal, y deja los resultados en `results/`:

1. Tests automáticos (`pytest`, 81 tests).
2. Patch test de los 4 elementos.
3. Ejemplo de verificación: viga en voladizo frente a la teoría de vigas.
4. Malla de la presa.
5. Presa con base empotrada: cálculo manual vs FEM, momentos y reacciones.
6. Presa con base Robin (fundación elástica).
7. Estudio comparativo: elementos, refinamiento y condiciones de borde.
8. Variaciones de geometría para aumentar el factor de seguridad.

Por defecto va **paso a paso**: empieza con un saludo y, antes de cada paso,
explica qué hace y espera: [Enter] ejecutar, [s] saltear, [q] salir. Los
gráficos se abren en ventanas (cerrarlas para continuar) y al terminar cada
paso muestra su resultado clave. Al final, un resumen y la opción de abrir el
informe. Antes de empezar verifica que estén instaladas las librerías.

```
python ejecutar_todo.py --automatico   # todo seguido, sin pausas ni ventanas
python ejecutar_todo.py --rapido       # saltea los estudios más largos
python ejecutar_todo.py --paso 5       # solo el paso 5
python ejecutar_todo.py --desde 5      # del paso 5 en adelante
python ejecutar_todo.py --sin-color    # sin colores en la terminal
```

## Qué hay en este repositorio

```
ejecutar_todo.py     ejecuta todo el trabajo en orden (punto de entrada)
informe/             informe final en PDF
  resultados/        reporte de la presa y figuras usadas en el informe
scripts/             scripts de cada paso (los llama ejecutar_todo.py)
  patch_test.py          patch test de los 4 elementos
  ejemplo_viga.py        viga en voladizo: verificación y visualización
  generar_malla_presa.py malla de la presa
  analisis_presa.py      presa: cálculo manual, FEM, momentos, apoyos
  estudio_geometria.py   variaciones de geometría y factor de seguridad
  bar.py, bar_roof.py    ejemplos de barras de la clase (código base)
source/              código de elementos finitos
meshes/              mallas en el formato del TP
tests/               tests automáticos (pytest)
requirements.txt     librerías necesarias
```

Cada script se puede correr por separado desde la raíz del repositorio, por
ejemplo `python scripts/analisis_presa.py --support robin`. Los resultados que
generan quedan en `results/` (no se sube al repositorio).

## Elementos disponibles

| Familia | Orden | Nodos | Nombre | Integración |
|---------|-------|-------|--------|-------------|
| TRI     | 1     | 3     | Tri3   | 1 punto     |
| TRI     | 2     | 6     | Tri6   | 3 puntos    |
| QUAD    | 1     | 4     | Quad4  | 2 x 2       |
| QUAD    | 2     | 9     | Q9     | 3 x 3       |

Todos pasan el patch test (`python scripts/patch_test.py`).

## Estructura

```
source/
  elements/      elementos finitos (Element2D, Tri3, Quad4, Tri6, Quad9)
  materials/     modelos de material
  mesh/          nodos, lectura y escritura de mallas, generadores
  quadrature/    reglas de integración de Gauss
  systems/       matriz global y condiciones de borde
  solvers/       resolución del sistema lineal
  output/        post-proceso, exportación a ParaView y gráficos
  verification/  patch test (por desplazamientos y por fuerzas)
meshes/          archivos de malla (presa/: mallas de la presa)
tests/           tests automáticos (pytest)
```

## Requisitos

```
pip install -r requirements.txt
```

## Uso

```
python scripts/patch_test.py     # patch test de los 4 elementos (versiones A y B)
python -m pytest         # todos los tests
python scripts/ejemplo_viga.py   # viga en voladizo: comparación con teoría y deformada
python scripts/generar_malla_presa.py   # malla de la presa (ver abajo)
python scripts/analisis_presa.py        # análisis de la presa (ver abajo)
python scripts/estudio_geometria.py     # variaciones de geometría (ver abajo)
python scripts/bar.py            # ejemplos de barras de la clase
python scripts/bar_roof.py
```

## Visualización

La solución se puede ver sobre la configuración deformada, con un factor de
escala elegible, de dos formas. Ejemplo con la viga:

```
python scripts/ejemplo_viga.py --scale 50 --field sigma_1
```

1. **matplotlib** (`source/output/plot.py`): `plot_deformed(mesh, elements, u,
   field, scale)` dibuja la malla deformada (desplazamientos x `scale`)
   coloreada con el campo elegido, sobre la malla original en gris. Si no se
   da `scale`, se elige uno automático (desplazamiento máximo = 5% del modelo).
   Campos: `sigma_1`, `sigma_2` (principales en el plano), `sigma_xx`,
   `sigma_yy`, `tau_xy`, `sigma_zz` (fuera del plano), `von_mises`,
   `displacement`. Los campos con signo usan una escala divergente centrada en
   cero (rojo tracción, azul compresión).

2. **ParaView** (`source/output/paraview.py`): `write_solid_results` escribe un
   `.vtu` con los desplazamientos y todas las tensiones en los nodos. En
   ParaView: abrir el archivo, aplicar el filtro **Warp By Vector** con el
   vector `displacement` y elegir el factor de escala; colorear con el campo
   que se quiera.

Las tensiones nodales se obtienen evaluando la tensión de cada elemento en sus
nodos y promediando entre los elementos que comparten cada nodo.

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

## Malla de la presa

`generar_malla_presa.py` genera la sección de la presa (H = 30 m, base 18 m,
cresta 3 m, borde libre 2 m) y la escribe en el formato de malla:

```
python scripts/generar_malla_presa.py                          # Q9, elementos de ~1 m
python scripts/generar_malla_presa.py --element TRI6 --size 0.5
python scripts/generar_malla_presa.py --batter 3               # talud aguas arriba
```

La sección es un cuadrilátero y se malla con un mapeo bilineal
(`source/mesh/generators.py`). Las filas de nodos son horizontales y una de
ellas cae exactamente en el nivel del agua (28 m), para que la presión
hidrostática se integre sin error. La malla incluye:

- NodeSets: `base`, `heel` (talón aguas arriba), `toe` (pie aguas abajo),
  `upstream`, `downstream`, `crest`.
- SideSets: `base`, `upstream`, `upstream_wet` (bajo el agua), `downstream`,
  `crest`.

En `meshes/presa/` están las mallas por defecto para los 4 tipos de elemento.

## Análisis de la presa

```
python scripts/analisis_presa.py                       # base empotrada, malla Q9
python scripts/analisis_presa.py --support roller      # rodillo (sin fricción)
python scripts/analisis_presa.py --support robin       # Robin: fundación elástica
python scripts/analisis_presa.py --comparar            # estudio completo
```

1. **Cálculo manual** (`source/dam/stability.py`): peso, empuje, momento
   estabilizante y desestabilizante respecto del pie, factor de seguridad al
   vuelco, posición de la resultante (tercio central) y ley trapecial de
   tensiones en la base. Suelo impermeable: sin subpresión.
2. **FEM** (`source/dam/model.py`): peso propio + presión hidrostática sobre
   `upstream_wet`, con una de tres condiciones de borde en la base:
   - `fixed`: empotrada, u = v = 0 (fundación rígida, fricción infinita);
   - `roller`: rodillo sin fricción, v = 0 y u = 0 solo en el talón;
   - `robin`: resortes distribuidos (condición de Robin, fundación elástica
     de Winkler), con k_n = E_f / B y k_t = G_f / B. Por defecto
     E_f = 20 GPa (`--e-fundacion`).

   La fricción de Coulomb es no lineal: `fixed` y `roller` son los dos casos
   límite del modelo lineal. Se comparan las reacciones y su momento
   respecto del pie con el cálculo manual.
3. **Figuras**: tensión principal máxima sobre la deformada y tensiones
   normales en la base (FEM vs ley trapecial); `.vtu` para ParaView.
4. **Estudio** (`--comparar`): 4 tipos de elemento, refinamiento de malla,
   condiciones de borde y deformación plana (adoptada) vs tensión plana.

Las tensiones se comparan en puntos fijos (`stress_at_point`) para que las
mallas distintas sean comparables. En las esquinas de la base empotrada la
tensión es singular (crece al refinar), así que la escala de colores se
satura con los valores a más de 1 m de esas esquinas.

El reporte queda en `results/presa_resultados.txt` (con `--support roller` o
`robin`, en `results/presa_resultados_<apoyo>.txt`).

## Variaciones de geometría

```
python scripts/estudio_geometria.py            # cálculo manual + verificación FEM
python scripts/estudio_geometria.py --sin-fem  # solo cálculo manual
```

Recorre cuatro familias de variantes (`source/dam/study.py`): base más
ancha hacia aguas abajo, talón extendido aguas arriba, cresta más ancha y
talud aguas arriba con la misma base. Para cada una informa el factor de
seguridad al vuelco, la posición de la resultante (tercio central), las
presiones en la base, la fricción necesaria para no deslizar y, con FEM, el
momento de las reacciones, el desplazamiento de la cresta y la tensión en el
paramento. Compara las variantes con la misma cantidad de hormigón.

El reporte queda en `results/estudio_geometria.txt`.

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
1     ELASTIC_PLANE_STRAIN  21.0e9   0.2   2400.0
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
