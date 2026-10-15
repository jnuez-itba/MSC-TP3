# TP3 - Estado plano de tensión

30.53 Mecánica del Sólido Computacional - ITBA

Código de elementos finitos en Python para modelar el equilibrio estático de un
sólido elástico lineal en estado plano de tensión. Parte de la estructura vista
en clase para barras 2D (`bar.py`, `bar_roof.py`) y la extiende a elementos de
sólido 2D.

## Estructura

```
source/
  elements/     elementos finitos
  materials/    modelos de material
  mesh/         nodos y lectura de mallas
  systems/      matriz global y condiciones de borde
  solvers/      resolución del sistema lineal
  output/       exportación de resultados (ParaView)
meshes/         archivos de malla
```

## Requisitos

```
pip install -r requirements.txt
```

## Uso

Ejemplos de barras de la clase:

```
python bar.py
python bar_roof.py
```
