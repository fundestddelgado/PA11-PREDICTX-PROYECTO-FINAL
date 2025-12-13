# PA11-PREDICTX-PROYECTO-FINAL
REPOSITORIO PARA EL PROYECTO FINAL Y HACKATON
# Sistema de Decisión para Retención de Clientes (Churn) con Explicabilidad

## Planteamiento
Desarrollar un modelo de machine learning que prediga la probabilidad de deserción (churn) de clientes bancarios para identificar tempranamente a los clientes con mayor riesgo y permitir estrategias preventivas de retención, incorporando explicabilidad para entender por qué el modelo toma cada decisión.

## Objetivos
- Predecir la probabilidad de churn por cliente y clasificar el riesgo (bajo/medio/alto).
- Evaluar el desempeño del modelo con métricas de clasificación.
- Explicar predicciones a nivel global y por cliente (factores que más influyen).
- Presentar resultados en un dashboard simple para consulta rápida.

## Herramientas utilizadas
- **Python**
- **Pycharm**: IDE
- **Jupyter Notebook** : pruebas y errores
- **Pandas / NumPy**: carga y preparación de datos.
- **Scikit-learn**: entrenamiento, validación y métricas (modelo principal: **Random Forest**).
- **SHAP**: explicabilidad global y local (importancia de variables, waterfall/summary).
- **Matplotlib** : visualizaciones.
- **Joblib/Pickle**: guardado y carga del modelo entrenado.
- **Dashboard web** : interfaz usando Streamlit con estilos (CSS).

## Resultado del proyecto
Se obtuvo un modelo entrenado capaz de identificar clientes con mayor riesgo de churn con un desempeño equilibrado en metricas, 
Con **SHAP** se identifican las variables que más influyen en el churn a nivel general y también se generan explicaciones por cliente para apoyar decisiones de retención.
