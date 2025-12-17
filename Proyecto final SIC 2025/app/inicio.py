from pathlib import Path
import sys

import streamlit as st
import seaborn as sns
import shap

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from carga.carga_y_datos import cargar_csv, limpiar_dataframe, cargar_modelo
from app.vistas import render_resumen, render_cliente

st.set_page_config(
    page_title="Dashboard Churn Bancario",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos seaborn + SHAP
sns.set_style("whitegrid")
shap.initjs()

# Cargar y añadir CSS
ruta_css = RAIZ / "estilos" / "estilos.css"
css = ruta_css.read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Rutas
RUTA_CSV = RAIZ / "datos" / "Churn_Modelling_sintetico_27500.csv"
RUTA_MODELO = RAIZ / "modelo" / "modelo_churn_rf_pipeline.joblib"

# Carga datos y modelo
try:
    df_raw = cargar_csv(RUTA_CSV)
    df_clean = limpiar_dataframe(df_raw)
except Exception as e:
    st.error(f" Error cargando el dataset: {e}")
    st.stop()

try:
    modelo = cargar_modelo(RUTA_MODELO)
except Exception as e:
    st.error(f" Error cargando el modelo: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ")
    vista = st.radio(
        "Secciones",
        ("Resumen general", " Análisis por cliente"),
    )

# Router
if "Resumen general" in vista:
    render_resumen(df_raw=df_raw, df_clean=df_clean, modelo=modelo)
else:
    render_cliente(df_raw=df_raw, df_clean=df_clean, modelo=modelo)
