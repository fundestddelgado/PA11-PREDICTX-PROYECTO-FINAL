import pandas as pd
import streamlit as st

def render_cliente(scored: pd.DataFrame, idcol: str):
    st.subheader("Detalle por cliente")

    cliente_id = st.selectbox(idcol, scored[idcol].astype(str).tolist())
    fila = scored[scored[idcol].astype(str) == str(cliente_id)].iloc[0]

    k1, k2, k3 = st.columns(3)
    k1.metric("Probabilidad churn", f"{fila['proba_churn']:.4f}")
    k2.metric("Predicción (umbral)", int(fila["pred_churn"]))
    k3.metric("Top 10% riesgo", "Sí" if int(fila["top10_flag"]) == 1 else "No")

    st.caption("aquí agregamos 'explicabilidad.")
