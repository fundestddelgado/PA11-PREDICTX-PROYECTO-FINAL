from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from general import render_general
from Lista_riesgo import render_lista_riesgo
from Cliente import render_cliente

# rutas
TARGET = "Churn"
IDCOL = "CustomerID"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datos"
MODEL_PATH = ROOT / "modelos" / "Modelo_RF.joblib"
OUT_DIR = ROOT / "salidas"
OUT_DIR.mkdir(exist_ok=True)


@st.cache_resource
def load_model(path: str):
    art = joblib.load(path)
    if isinstance(art, dict) and "pipeline" in art:
        pipeline = art["pipeline"]
        threshold = float(art.get("threshold", 0.50))
    else:
        pipeline = art
        threshold = 0.50
    return pipeline, threshold

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def score_df(df: pd.DataFrame, pipeline, threshold: float) -> pd.DataFrame:
    if IDCOL not in df.columns:
        raise ValueError(f"El CSV debe tener la columna '{IDCOL}'.")

    ids = df[IDCOL].copy()

    drop_cols = [IDCOL]
    if TARGET in df.columns:
        drop_cols.append(TARGET)

    X = df.drop(columns=drop_cols)

    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        IDCOL: ids,
        "proba_churn": proba,
        "pred_churn": pred
    }).sort_values("proba_churn", ascending=False).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)

    k = max(1, int(0.10 * len(out)))
    out["top10_flag"] = 0
    out.loc[:k-1, "top10_flag"] = 1
    return out

# =========================
# UI
# =========================
st.set_page_config(page_title="Plataforma", layout="wide")
st.title("Plataforma de Alerta Temprana de Churn (Prototipo)")

# Validaciones
if not MODEL_PATH.exists():
    st.error(f"No existe el modelo: {MODEL_PATH}")
    st.stop()
if not DATA_DIR.exists():
    st.error(f"No existe la carpeta de datos: {DATA_DIR}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Datos")
    precargados = []
    for name in ["train.csv", "test.csv"]:
        if (DATA_DIR / name).exists():
            precargados.append(name)

    if not precargados:
        st.error("No encontré train.csv ni test.csv dentro de /datos")
        st.stop()

    elegido = st.selectbox("Dataset precargado", precargados, index=0)
    up = st.file_uploader("O subir un CSV", type=["csv"])
    recalcular = st.button("Recalcular")

# Cargar modelo
pipeline, threshold = load_model(str(MODEL_PATH))
st.sidebar.success(f"Modelo cargado ✓  | Umbral: {threshold:.2f}")

# Cargar dataset
if up is not None:
    df = pd.read_csv(up)
    fuente = "CSV subido"
else:
    df = load_csv(str(DATA_DIR / elegido))
    fuente = f"Precargado: {elegido}"

# Scoring automático
src_key = (fuente, len(df), tuple(df.columns))
if recalcular or ("scored" not in st.session_state) or (st.session_state.get("src_key") != src_key):
    try:
        scored = score_df(df, pipeline, threshold)
        st.session_state["df"] = df
        st.session_state["scored"] = scored
        st.session_state["src_key"] = src_key

        scored.to_csv(OUT_DIR / "scoring.csv", index=False)
        scored[scored["top10_flag"] == 1].to_csv(OUT_DIR / "lista_top10.csv", index=False)
    except Exception as e:
        st.error(f"Error al calcular scoring: {e}")
        st.stop()

df = st.session_state["df"]
scored = st.session_state["scored"]

# Tabs
tab1, tab2, tab3 = st.tabs(["Resumen", "Lista de acción", "Detalle cliente"])

with tab1:
    render_general(df=df, scored=scored, fuente=fuente, target=TARGET)

with tab2:
    render_lista_riesgo(scored=scored)

with tab3:
    render_cliente(scored=scored, idcol=IDCOL)
