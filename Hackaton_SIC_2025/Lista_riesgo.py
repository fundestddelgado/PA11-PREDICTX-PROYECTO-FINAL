import pandas as pd
import streamlit as st

def render_lista_riesgo(scored: pd.DataFrame):
    st.subheader("Ranking de clientes por riesgo")
    solo_top10 = st.toggle("Mostrar solo Top 10%", value=False)

    view = scored.copy()
    if solo_top10:
        view = view[view["top10_flag"] == 1]

    st.dataframe(view, use_container_width=True, height=520)

    st.download_button(
        "Descargar CSV (vista actual)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="lista_accion.csv",
        mime="text/csv"
    )
