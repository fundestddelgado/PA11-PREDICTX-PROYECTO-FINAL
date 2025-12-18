import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Utilidades
# -------------------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


@st.cache_data
def _thresholds(df: pd.DataFrame):

    th = {}

    # Candidatos típicos
    cols = {
        "hours": ["ViewingHoursPerWeek"],
        "duration": ["AverageViewingDuration"],
        "downloads": ["ContentDownloadsPerMonth", "ContentDowloadsPerMonth"],
        "tickets": ["SupportTicketsPerMonth", "SupportTickets", "Tickets", "NumTickets"],
        "tenure": ["AccountAge", "Tenure"],
        "totalcharges": ["TotalCharges"],
        "monthly": ["MonthlyCharges", "MonthlyCharge", "PricePerMonth"]
    }

    def pick(cands):
        return next((c for c in cands if c in df.columns), None)

    for k, cands in cols.items():
        col = pick(cands)
        if col is None:
            th[k] = {"col": None}
            continue

        x = _to_num(df[col])
        th[k] = {
            "col": col,
            "p25": float(x.quantile(0.25)) if x.notna().sum() else None,
            "p50": float(x.quantile(0.50)) if x.notna().sum() else None,
            "p75": float(x.quantile(0.75)) if x.notna().sum() else None,
        }

    return th


def _safe_get(row: pd.Series, col: str):
    if col is None or col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    return v


# -------------------------
# Reglas por motivo (1 función = 1 motivo)
# -------------------------
def motivo_bajo_uso(cliente_row: pd.Series, th: dict):
    """
    Bajo uso: horas/duración/descargas en cuartil bajo.
    """
    hours_col = th["hours"]["col"]
    dur_col = th["duration"]["col"]
    down_col = th["downloads"]["col"]

    señales = []

    if hours_col and th["hours"]["p25"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, hours_col)])).iloc[0]
        if pd.notna(v) and v <= th["hours"]["p25"]:
            señales.append(f"{hours_col} bajo")

    if dur_col and th["duration"]["p25"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, dur_col)])).iloc[0]
        if pd.notna(v) and v <= th["duration"]["p25"]:
            señales.append(f"{dur_col} bajo")

    if down_col and th["downloads"]["p25"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, down_col)])).iloc[0]
        if pd.notna(v) and v <= th["downloads"]["p25"]:
            señales.append(f"{down_col} bajo")

    if len(señales) == 0:
        return None

    return {
        "motivo": "Bajo uso (desenganche)",
        "señales": señales,
        "accion": "Campaña de activación: recomendaciones personalizadas, recordatorios in-app/email, y guía rápida de valor.",
        "canal": "In-app + Email",
        "prioridad": "P1" if len(señales) >= 2 else "P2"
    }


def motivo_alta_friccion_soporte(cliente_row: pd.Series, th: dict):
    """
    Fricción alta: tickets en cuartil alto.
    """
    tix_col = th["tickets"]["col"]
    if not tix_col or th["tickets"]["p75"] is None:
        return None

    v = _to_num(pd.Series([_safe_get(cliente_row, tix_col)])).iloc[0]
    if pd.isna(v):
        return None

    if v >= th["tickets"]["p75"]:
        return {
            "motivo": "Alta fricción (soporte/tickets)",
            "señales": [f"{tix_col} alto"],
            "accion": "Soporte proactivo: contacto humano, resolución de incidencias y seguimiento 48h.",
            "canal": "Soporte / WhatsApp / Email",
            "prioridad": "P1"
        }
    return None


def motivo_churn_temprano_onboarding(cliente_row: pd.Series, th: dict):
    """
    Churn temprano: antigüedad/tenure bajo.
    """
    ten_col = th["tenure"]["col"]
    if not ten_col or th["tenure"]["p25"] is None:
        return None

    v = _to_num(pd.Series([_safe_get(cliente_row, ten_col)])).iloc[0]
    if pd.isna(v):
        return None

    if v <= th["tenure"]["p25"]:
        return {
            "motivo": "Riesgo temprano (onboarding)",
            "señales": [f"{ten_col} bajo"],
            "accion": "Onboarding guiado: checklist de primeros pasos + recomendaciones iniciales + nudges de activación.",
            "canal": "In-app + Email",
            "prioridad": "P2"
        }
    return None


def motivo_sensible_precio(cliente_row: pd.Series, th: dict):
    """
    Sensible a precio (aprox): TotalCharges bajo o MonthlyCharges alto (si existe).
    """
    tc_col = th["totalcharges"]["col"]
    m_col = th["monthly"]["col"]

    señales = []

    if tc_col and th["totalcharges"]["p25"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, tc_col)])).iloc[0]
        if pd.notna(v) and v <= th["totalcharges"]["p25"]:
            señales.append(f"{tc_col} bajo")

    if m_col and th["monthly"]["p75"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, m_col)])).iloc[0]
        if pd.notna(v) and v >= th["monthly"]["p75"]:
            señales.append(f"{m_col} alto")

    if not señales:
        return None

    return {
        "motivo": "Sensibilidad a precio / valor percibido",
        "señales": señales,
        "accion": "Ofrecer ajuste de plan: descuento temporal, downgrade sugerido o bundle, reforzando beneficios clave.",
        "canal": "Email + In-app",
        "prioridad": "P2"
    }


def motivo_alto_valor(cliente_row: pd.Series, th: dict):
    """
    Alto valor: TotalCharges en cuartil alto (o MonthlyCharges alto).
    """
    tc_col = th["totalcharges"]["col"]
    m_col = th["monthly"]["col"]

    señales = []

    if tc_col and th["totalcharges"]["p75"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, tc_col)])).iloc[0]
        if pd.notna(v) and v >= th["totalcharges"]["p75"]:
            señales.append(f"{tc_col} alto")

    if m_col and th["monthly"]["p75"] is not None:
        v = _to_num(pd.Series([_safe_get(cliente_row, m_col)])).iloc[0]
        if pd.notna(v) and v >= th["monthly"]["p75"]:
            señales.append(f"{m_col} alto")

    if not señales:
        return None

    return {
        "motivo": "Cliente de alto valor (priorizar retención)",
        "señales": señales,
        "accion": "Retención premium: contacto prioritario, beneficio personalizado y soporte preferencial.",
        "canal": "Soporte / Llamada / Email",
        "prioridad": "P1"
    }


# -------------------------
# Motor de intervención (solo Top10)
# -------------------------
def generar_intervenciones(df: pd.DataFrame, cliente_row: pd.Series):
    th = _thresholds(df)

    reglas = [
        motivo_alto_valor,
        motivo_alta_friccion_soporte,
        motivo_bajo_uso,
        motivo_churn_temprano_onboarding,
        motivo_sensible_precio,
    ]

    recs = []
    for regla in reglas:
        out = regla(cliente_row, th)
        if out is not None:
            recs.append(out)

    # Orden simple por prioridad
    prio_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    recs.sort(key=lambda x: prio_order.get(x["prioridad"], 99))
    return recs


def render_intervencion_cliente(df: pd.DataFrame, cliente_row: pd.Series, fila_score: pd.Series):
    st.subheader("Intervención recomendada")

    # SOLO si está en Top 10%
    if int(fila_score.get("top10_flag", 0)) != 1:
        st.info("Este cliente no está en el Top 10% de riesgo. No se generan recomendaciones.")
        return

    recs = generar_intervenciones(df, cliente_row)

    if not recs:
        st.warning("Top 10% detectado, pero no se activaron reglas con las columnas disponibles.")
        return

    # Mostrar recomendaciones
    for i, r in enumerate(recs, start=1):
        with st.expander(f"{r['prioridad']} — {r['motivo']}", expanded=(i == 1)):
            st.write("**Señales:** " + ", ".join(r["señales"]))
            st.write("**Acción:** " + r["accion"])
            st.write("**Canal sugerido:** " + r["canal"])

render_intervencion = render_intervencion_cliente