import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportes import build_cliente_report_pdf
from intervencion import render_intervencion, generar_intervenciones
import shap


# =========================
# KPI Cards (usan tu CSS externo)
# =========================
def _kpi_card(titulo: str, valor: str, subtitulo: str = ""):
    sub_html = f'<div class="kpi-sub">{subtitulo}</div>' if subtitulo else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{titulo}</div>
            <div class="kpi-value">{valor}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def _get_explainer(_rf_model):
    return shap.TreeExplainer(_rf_model)


def _get_cat_cols(preprocess):
    cat_cols = []
    for name, trans, cols in getattr(preprocess, "transformers_", []):
        if name == "cat":
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                cat_cols = list(cols)
            break
    return cat_cols


def _prettify_feature_names(feat_names, preprocess):
    """
    Convierte nombres del preprocesamiento a nombres legibles:
    - num__TotalCharges -> TotalCharges
    - cat__SubscriptionType_Premium -> SubscriptionType=Premium
    """
    cat_cols = _get_cat_cols(preprocess)
    pretty = []

    for fn in feat_names:
        fn = str(fn)

        if fn.startswith("num__"):
            pretty.append(fn.replace("num__", "", 1))
            continue

        if fn.startswith("cat__"):
            rest = fn.replace("cat__", "", 1)

            # Intentar mapear usando las columnas categóricas originales
            mapped = None
            for col in cat_cols:
                prefix = f"{col}_"
                if rest.startswith(prefix):
                    cat_val = rest[len(prefix):]
                    mapped = f"{col}={cat_val}"
                    break

            pretty.append(mapped if mapped is not None else rest)
            continue

        # fallback
        pretty.append(fn)

    return np.array(pretty, dtype=object)


def _shap_local_class1(pipeline, X_row: pd.DataFrame):
    """
    Devuelve (pretty_feature_names, shap_values_1d_class1, base_value_class1, X_trans_row_1d)
    """
    if shap is None:
        return None, "SHAP no está instalado. Agrega 'shap' a requirements.txt."

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_t = preprocess.transform(X_row)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    # nombres de features transformadas
    try:
        feat_names = preprocess.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f_{i}" for i in range(X_t.shape[1])])

    # nombres “humanos”
    pretty_names = _prettify_feature_names(feat_names, preprocess)

    explainer = _get_explainer(model)
    sv = explainer.shap_values(X_t)

    # --- Normalizar a vector 1D para clase 1 ---
    if isinstance(sv, list):
        vals = np.asarray(sv[1])[0] if len(sv) > 1 else np.asarray(sv[0])[0]
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:
            cls_idx = 1 if sv.shape[2] > 1 else 0
            vals = sv[0, :, cls_idx]
        elif sv.ndim == 2:
            vals = sv[0]
        else:
            vals = sv.ravel()

    vals = np.asarray(vals).ravel()

    # base value para clase 1
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and len(np.asarray(ev).ravel()) > 1:
        base = float(np.asarray(ev).ravel()[1])
    else:
        base = float(np.asarray(ev).ravel()[0])

    # seguridad por si no coincide longitud
    m = min(len(vals), len(pretty_names), X_t.shape[1])
    vals = vals[:m]
    pretty_names = pretty_names[:m]
    X_t = X_t[:, :m]

    return (pretty_names, vals, base, X_t[0]), None


def _plot_shap_top5_waterfall(feature_names, vals, base, x_row):
    """
    Waterfall Top 5 usando SHAP + matplotlib.
    """
    exp = shap.Explanation(
        values=vals,
        base_values=base,
        data=x_row,
        feature_names=feature_names
    )

    plt.figure()
    try:
        shap.plots.waterfall(exp, max_display=5, show=False)
    except Exception:
        # fallback compatibilidad
        shap.waterfall_plot(exp.base_values, exp.values, exp.data,
                            feature_names=exp.feature_names, max_display=5, show=False)

    fig = plt.gcf()
    return fig


def render_cliente(df: pd.DataFrame, scored: pd.DataFrame, idcol: str, target: str, pipeline):
    st.subheader("Detalle por cliente")

    # Selector
    ids = scored[idcol].astype(str).tolist()
    ids = ids[10:]  # quitar los 10 primeros
    colA, colB = st.columns([2, 1])

    # Top 10 IDs (de mayor riesgo dentro del top10%)
    top_ids = (
        scored[scored["top10_flag"] == 1]
        .sort_values("proba_churn", ascending=False)[idcol]
        .astype(str)
        .head(10)
        .tolist()
    )

    # Estado inicial
    if "syncing_select" not in st.session_state:
        st.session_state["syncing_select"] = False

    if "general_id" not in st.session_state:
        st.session_state["general_id"] = ids[0] if ids else ""

    if "top_risk_choice" not in st.session_state:
        st.session_state["top_risk_choice"] = "(Selecciona un ID)"

    def on_general_change():
        # Si el cambio viene por sincronización interna, no resetees
        if st.session_state.get("syncing_select"):
            return
        # Si el usuario usa búsqueda general, resetea top riesgo
        st.session_state["top_risk_choice"] = "(Selecciona un ID)"

    def on_top_change():
        choice = st.session_state.get("top_risk_choice")
        if choice and choice != "(Selecciona un ID)":
            # Si el usuario usa top riesgo, sincroniza el general al mismo ID
            st.session_state["syncing_select"] = True
            st.session_state["general_id"] = choice
            st.session_state["syncing_select"] = False
        # (Opcional) si vuelve a placeholder, no hacemos nada

    colA, colB = st.columns([2, 1])

    with colA:
        st.selectbox(
            "Buscar ID general",
            ids,
            key="general_id",
            on_change=on_general_change
        )

    with colB:
        st.selectbox(
            "Top riesgo",
            ["(Selecciona un ID)"] + top_ids,
            key="top_risk_choice",
            on_change=on_top_change
        )

    # El ID final siempre viene del general (porque top sincroniza general)
    cliente_id = st.session_state["general_id"]
    fila_score = scored[scored[idcol].astype(str) == str(cliente_id)].iloc[0]

    # KPIs (tarjetas)
    prob = float(fila_score["proba_churn"])
    pred = int(fila_score["pred_churn"])
    top10 = int(fila_score["top10_flag"]) == 1
    rank = int(fila_score.get("rank", np.nan)) if "rank" in fila_score else None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card("Probabilidad churn", f"{prob:.4f}", "Score del modelo")
    with k2:
        _kpi_card("Predicción", f"{pred}", "1=Churn, 0=No churn")
    with k3:
        _kpi_card("Top 10% riesgo", "Sí" if top10 else "No", "Prioridad")
    with k4:
        _kpi_card("Ranking riesgo", f"{rank}" if rank else "N/A", "1 = mayor riesgo")

    # Perfil
    # Perfil + SHAP en columnas
    if idcol not in df.columns:
        st.info(f"El dataset base no trae '{idcol}'.")
        return

    cliente_rows = df[df[idcol].astype(str) == str(cliente_id)]
    if cliente_rows.empty:
        st.info("No encontré este ID en el dataset base.")
        return

    cliente_row = cliente_rows.iloc[0]

    # --- Preparar X para SHAP ---
    drop_cols = [idcol]
    if target in df.columns:
        drop_cols.append(target)
    X_row = cliente_rows.drop(columns=drop_cols).head(1)

    out, err = _shap_local_class1(pipeline, X_row)
    if err:
        st.info(err)
        return

    feature_names, vals, base, x_row = out
    fig = _plot_shap_top5_waterfall(feature_names, vals, base, x_row)

    # --- Tabla vertical (Variable | Valor) ---
    excluir = {target}  # no mostrar churn
    orden = [idcol] + [c for c in cliente_row.index if c not in excluir and c != idcol]
    cliente_row_ord = cliente_row[orden]

    perfil_df = (
        cliente_row_ord.to_frame(name="Valor")
        .reset_index()
        .rename(columns={"index": "Variable"})
    )

    colL, colR = st.columns([0.95, 1.25])

    with colL:
        st.subheader("Perfil del cliente")
        st.dataframe(perfil_df, use_container_width=True, height=260)

    with colR:
        st.subheader("Impacto según el modelo")
        st.pyplot(fig, use_container_width=True, clear_figure=True)

    st.caption("Valores SHAP positivos empujan hacia churn=1; negativos hacia churn=0.")

    #st.subheader("Intervención sugerida")
    render_intervencion(df=df, cliente_row=cliente_row, fila_score=fila_score)

    # Top 5 SHAP (para el informe)
    idx = np.argsort(np.abs(vals))[::-1][:5]
    shap_top5 = [(feature_names[i], float(vals[i])) for i in idx]

    # Recomendaciones SOLO si Top10
    recs = []
    if int(fila_score.get("top10_flag", 0)) == 1:
        recs = generar_intervenciones(df, cliente_row)

    pdf_cliente = build_cliente_report_pdf(
        cliente_id=str(cliente_id),
        fila_score=fila_score,
        cliente_row=cliente_row,
        shap_top5=shap_top5,
        recomendaciones=recs,
        target=target
    )

    st.download_button(
        "Descargar informe",
        data=pdf_cliente,
        file_name=f"informe_cliente_{cliente_id}.pdf",
        mime="application/pdf"
    )

