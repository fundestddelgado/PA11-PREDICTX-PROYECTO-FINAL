import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from reportes import build_general_report_pdf

# Azul oscuro fijo (puedes cambiarlo si quieres)
DARK_BLUE = "#0B3D91"


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


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _bar_simple(plot_df: pd.DataFrame, x_col: str, y_col: str = "churn_pct"):
    """Barra simple azul oscuro, sin leyenda ni color por condición."""
    if plot_df.empty:
        return None

    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        labels={y_col: "Churn (%)", x_col: ""},
        title=""
    )
    fig.update_traces(
        marker_color=DARK_BLUE,
        hovertemplate=f"{x_col}=%{{x}}<br>Churn=%{{y:.2f}}%<extra></extra>"
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    return fig

def fig_usabilidad_churn_por_categoria(df: pd.DataFrame, target: str):
    if target not in df.columns:
        return None, "El dataset no trae 'Churn' (no se puede calcular % churn por categorías)."

    # Columnas de usabilidad (con typo incluido)
    features = []
    for c in ["ViewingHoursPerWeek", "AverageViewingDuration", "ContentDownloadsPerMonth", "ContentDowloadsPerMonth"]:
        if c in df.columns:
            features.append(c)

    # Evitar duplicado si están las dos
    if "ContentDownloadsPerMonth" in features and "ContentDowloadsPerMonth" in features:
        features.remove("ContentDowloadsPerMonth")

    if not features:
        return None, "No encontré columnas de usabilidad (ViewingHoursPerWeek, AverageViewingDuration, ContentDownloadsPerMonth)."

    tmp = df[features + [target]].copy()
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna(subset=[target])
    tmp[target] = tmp[target].astype(int)

    # Convertir usabilidad a numérico
    for f in features:
        tmp[f] = pd.to_numeric(tmp[f], errors="coerce")

    # Categorías por cuantiles (Bajo/Medio/Alto) para cada feature
    cat_labels = ["Bajo", "Medio", "Alto"]
    long_rows = []

    for f in features:
        s = tmp[f].copy()
        # si hay muchos NaN o valores constantes, no sirve categorizar
        if s.dropna().nunique() < 3:
            continue

        # qcut crea 3 grupos con tamaños parecidos
        cats = pd.qcut(s, q=3, labels=cat_labels, duplicates="drop")
        aux = pd.DataFrame({"cat": cats, "y": tmp[target]})
        aux = aux.dropna(subset=["cat", "y"])

        churn_pct = aux.groupby("cat")["y"].mean() * 100
        counts = aux.groupby("cat")["y"].size()

        for c in cat_labels:
            if c in churn_pct.index:
                long_rows.append({
                    "Categoria": c,
                    "Metrica": f,
                    "ChurnPct": float(churn_pct.loc[c]),
                    "n": int(counts.loc[c])
                })

    plot_df = pd.DataFrame(long_rows)

    if plot_df.empty:
        return None, "No se pudo construir la gráfica (revisa si las columnas tienen suficiente variación)."

    # Orden fijo en el eje Y
    plot_df["Categoria"] = pd.Categorical(plot_df["Categoria"], categories=cat_labels, ordered=True)
    plot_df = plot_df.sort_values(["Categoria", "Metrica"])

    # Nombres más bonitos (opcional)
    rename_map = {
        "ViewingHoursPerWeek": "Horas/semana",
        "AverageViewingDuration": "Duración promedio",
        "ContentDownloadsPerMonth": "Descargas/mes",
        "ContentDowloadsPerMonth": "Descargas/mes"
    }
    plot_df["Metrica"] = plot_df["Metrica"].replace(rename_map)

    fig = px.bar(
        plot_df,
        x="ChurnPct",
        y="Categoria",
        color="Metrica",
        barmode="group",
        orientation="h",
        labels={"ChurnPct": "Porcentaje de Churn", "Categoria": "Usabilidad"},
        title=""
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title=""
    )

    fig.update_traces(
        hovertemplate="% Churn=%{x:.2f}%<br>Categoria=%{y}<br>n=%{customdata}<extra></extra>",
        customdata=plot_df["n"]
    )

    return fig, None



# =========================
# 1) Churn vs Antigüedad (binned)
# =========================
def fig_churn_antiguedad(df: pd.DataFrame, target: str):
    age_col = _find_col(df, ["AccountAge", "account_age", "Tenure", "tenure", "EdadCuenta", "edad_cuenta"])
    if (age_col is None) or (target not in df.columns):
        return None, "Falta AccountAge/Tenure y/o Churn."

    tmp = df[[age_col, target]].copy()
    tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna(subset=[age_col, target])

    bins = [0, 3, 6, 12, 24, 36, 60, np.inf]
    labels = ["0-3", "4-6", "7-12", "13-24", "25-36", "37-60", "60+"]

    tmp["intervalo"] = pd.cut(tmp[age_col], bins=bins, labels=labels, include_lowest=True)

    grp = tmp.groupby("intervalo")[target].mean().reindex(labels)
    plot_df = grp.reset_index().rename(columns={target: "churn_rate"})
    plot_df["churn_pct"] = plot_df["churn_rate"] * 100
    plot_df = plot_df.dropna(subset=["intervalo"])

    fig = _bar_simple(plot_df, "intervalo")
    return fig, None


# =========================
# 2) Churn vs Tipo de suscripción
# =========================
def fig_churn_suscripcion(df: pd.DataFrame, target: str):
    sub_col = _find_col(df, ["SubscriptionType", "subscription_type", "Plan", "PlanType", "plan_type", "TipoSuscripcion", "tipo_suscripcion"])
    if (sub_col is None) or (target not in df.columns):
        return None, "Falta SubscriptionType (o similar) y/o Churn."

    tmp = df[[sub_col, target]].copy().dropna(subset=[sub_col, target])
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna(subset=[target])

    grp = tmp.groupby(sub_col)[target].mean().sort_values(ascending=False)
    plot_df = grp.reset_index().rename(columns={target: "churn_rate"})
    plot_df["churn_pct"] = plot_df["churn_rate"] * 100

    fig = _bar_simple(plot_df, sub_col)
    return fig, None


# =========================
# 3) Churn vs Tickets (binned)
# =========================
def fig_churn_tickets(df: pd.DataFrame, target: str):
    tix_col = _find_col(df, ["SupportTicketsPerMonth", "SupportTickets", "Tickets", "tickets", "NumTickets", "num_tickets"])
    if (tix_col is None) or (target not in df.columns):
        return None, "Falta columna de tickets y/o Churn."

    tmp = df[[tix_col, target]].copy()
    tmp[tix_col] = pd.to_numeric(tmp[tix_col], errors="coerce")
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna(subset=[tix_col, target])

    bins_t = [-0.1, 0, 1, 2, 3, 5, np.inf]
    labels_t = ["0", "1", "2", "3", "4-5", "6+"]

    tmp["tickets"] = pd.cut(tmp[tix_col], bins=bins_t, labels=labels_t, include_lowest=True)

    grp = tmp.groupby("tickets")[target].mean().reindex(labels_t)
    plot_df = grp.reset_index().rename(columns={target: "churn_rate"})
    plot_df["churn_pct"] = plot_df["churn_rate"] * 100
    plot_df = plot_df.dropna(subset=["tickets"])

    fig = _bar_simple(plot_df, "tickets")
    return fig, None

def fig_ojiva_totalcharges_churn(df: pd.DataFrame, target: str, nbins: int = 20):
    if "TotalCharges" not in df.columns or target not in df.columns:
        return None, "Falta 'TotalCharges' y/o 'Churn'."

    tmp = df[["TotalCharges", target]].copy()
    tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
    tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
    tmp = tmp.dropna(subset=["TotalCharges", target])
    tmp[target] = tmp[target].astype(int)

    # recorte de outliers para que la curva no se distorsione
    lo = float(tmp["TotalCharges"].quantile(0.01))
    hi = float(tmp["TotalCharges"].quantile(0.99))
    if hi > lo:
        tmp = tmp[(tmp["TotalCharges"] >= lo) & (tmp["TotalCharges"] <= hi)]

    total_churners = int(tmp[target].sum())
    if total_churners == 0:
        return None, "No hay churners (Churn=1) en este dataset."

    # bins y agregación
    bins = np.linspace(tmp["TotalCharges"].min(), tmp["TotalCharges"].max(), nbins + 1)
    tmp["bin"] = pd.cut(tmp["TotalCharges"], bins=bins, include_lowest=True)

    grp = tmp.groupby("bin", observed=True)[target].agg(n="size", churners="sum").reset_index()
    grp["x"] = grp["bin"].apply(lambda iv: float(iv.right))  # tope del bin

    grp["cum_churners"] = grp["churners"].cumsum()
    grp["cum_churn_pct"] = 100 * grp["cum_churners"] / total_churners

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grp["x"],
        y=grp["cum_churn_pct"],
        mode="lines+markers",
        hovertemplate="TotalCharges≤%{x:.0f}<br>% Churn acumulado=%{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Total de cargo (umbral)",
        yaxis_title="% Churn acumulado",
    )
    fig.update_yaxes(range=[0, 100])
    return fig, None


# =========================
# Render principal
# =========================
def render_general(df: pd.DataFrame, scored: pd.DataFrame, fuente: str, target: str):
    st.subheader("Resumen general")

    total = len(df)
    if target in df.columns:
        y = df[target].astype(int)
        churners = int(y.sum())
        churn_rate = float(y.mean())
    else:
        churners, churn_rate = None, None

    col1, col2, col3 = st.columns(3)
    with col1:
        _kpi_card("Clientes totales", f"{total:,}", "Base cargada")
    with col2:
        _kpi_card("Clientes perdidos", f"{churners:,}" if churners is not None else "N/A", "Churn = 1" if churners is not None else "No viene Churn")
    with col3:
        _kpi_card("Porcentaje de churn", f"{100*churn_rate:.2f}%" if churn_rate is not None else "N/A", "Tasa global" if churn_rate is not None else "No viene Churn")

    st.caption(f"Fuente: {fuente}")

    #st.subheader("Churn por segmentos")
    colA, colB, colC = st.columns(3)

    with colA:
        st.caption("Antigüedad")
        fig, err = fig_churn_antiguedad(df, target)
        if err: st.info(err)
        else: st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.caption("Suscripción")
        fig, err = fig_churn_suscripcion(df, target)
        if err: st.info(err)
        else: st.plotly_chart(fig, use_container_width=True)

    with colC:
        st.caption("Tickets")
        fig, err = fig_churn_tickets(df, target)
        if err: st.info(err)
        else: st.plotly_chart(fig, use_container_width=True)

    #st.subheader("Usabilidad y TotalCharges")

    colL, colR = st.columns(2)

    with colL:
        st.caption("Churn con respecto a la Usabilidad")
        fig_u, err_u = fig_usabilidad_churn_por_categoria(df, target)
        if err_u:
            st.info(err_u)
        else:
            st.plotly_chart(fig_u, use_container_width=True)

    with colR:
        st.caption("churn acumulado vs Cargo Total")
        fig_o, err_o = fig_ojiva_totalcharges_churn(df, target, nbins=20)
        if err_o:
            st.info(err_o)
        else:
            st.plotly_chart(fig_o, use_container_width=True)

    pdf_general = build_general_report_pdf(df=df, scored=scored, fuente=fuente, target=target)
    st.download_button(
        "Descargar informe",
        data=pdf_general,
        file_name="informe_general.pdf",
        mime="application/pdf"
    )



