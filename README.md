# MILP-AHP Streamlit App

Aplicación en Streamlit para resolver un modelo de optimización de asignación de recursos en estrategias de muestreo de seeps de hidrógeno.

## Qué hace

- Lee el archivo `Data.xlsx` con parámetros de tiempo, costos, mínimos, pesos AHP y perímetros de círculos.
- Convierte los pesos AHP en proporciones objetivo de presupuesto.
- Resuelve el modelo en tres etapas lexicográficas:
  1. Minimiza desviación Campo/Laboratorio.
  2. Minimiza desviaciones internas por actividad y análisis.
  3. Maximiza presupuesto utilizado manteniendo las desviaciones previas.
- Permite editar actividades, costos, calidades AHP, separaciones y restricciones antes de ejecutar.
- Exporta resultados en CSV.

## Cómo correr localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cómo subir a Streamlit Cloud

1. Crea un repositorio en GitHub.
2. Sube `app.py`, `requirements.txt`, la carpeta `.streamlit/` y opcionalmente `sample_data/Data.xlsx`.
3. En Streamlit Cloud selecciona el repositorio y como archivo principal usa `app.py`.
4. Al abrir la app, carga tu propio `Data.xlsx` o usa el archivo de ejemplo incluido.

## Formato esperado del Excel

- Hoja `Master_Datos`: columnas tipo `Parámetro`, `Unidad`, `Valor`.
- Hoja `Sheet2`: columnas `ID` y `Perimetro (m)`.

## Nota del modelo

El solver usado es OR-Tools CP-SAT. Para mantener la app interactiva, cada etapa tiene un límite de tiempo configurable. Si el estado es `FEASIBLE`, significa que encontró una solución válida, aunque no necesariamente pudo demostrar optimalidad dentro del tiempo asignado.
