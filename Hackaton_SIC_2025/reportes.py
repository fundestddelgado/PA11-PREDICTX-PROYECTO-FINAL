from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.platypus import Image
from reportlab.lib.units import inch

def _fmt(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def _fig_to_rl_image(fig, width=6.8*inch):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth = width
    img.drawHeight = img.imageHeight * (width / img.imageWidth)
    return img


def _pick_col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def _safe_top(series):
    # series index=categoría, values=churn_pct
    s = series.dropna()
    if s.empty:
        return None, None
    idx = s.idxmax()
    return str(idx), float(s.loc[idx])

def build_general_report_pdf(df: pd.DataFrame, scored: pd.DataFrame, fuente: str, target: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Informe general — Plataforma de Alerta Temprana de Churn", styles["Title"]))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"Fuente dataset: {fuente}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ===== Resumen ejecutivo (igual que antes) =====
    total = len(df)
    churn_count = None
    churn_rate = None
    if target in df.columns:
        churn_rate = float(pd.to_numeric(df[target], errors="coerce").mean())
        churn_count = int(pd.to_numeric(df[target], errors="coerce").sum())

    avg_risk = float(scored["proba_churn"].mean()) if "proba_churn" in scored.columns else None
    top10_n = int(scored["top10_flag"].sum()) if "top10_flag" in scored.columns else None

    metrics = [
        ["Métrica", "Valor"],
        ["Clientes totales", f"{total:,}"],
        ["Clientes churn (si existe)", f"{churn_count:,}" if churn_count is not None else "N/A"],
        ["% churn (si existe)", f"{100*churn_rate:.2f}%" if churn_rate is not None else "N/A"],
        ["Riesgo promedio (modelo)", _fmt(avg_risk, 4)],
        ["Tamaño Top 10% (modelo)", f"{top10_n:,}" if top10_n is not None else "N/A"],
    ]
    t = Table(metrics, colWidths=[220, 300])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Paragraph("Resumen ", styles["Heading2"]))
    story.append(t)
    story.append(Spacer(1, 14))

    # ===== Gráficas + resultados =====
    story.append(Paragraph("Gráficas clave y hallazgos", styles["Heading2"]))

    if target not in df.columns:
        story.append(Paragraph("El dataset no contiene la columna 'Churn', por lo que no se pueden calcular % churn en segmentos.", styles["Normal"]))
        doc.build(story)
        return buf.getvalue()

    # ------------ 1) Churn por Antigüedad ------------
    age_col = _pick_col(df, ["AccountAge", "Tenure", "account_age", "tenure"])
    if age_col:
        tmp = df[[age_col, target]].copy()
        tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
        tmp[target] = pd.to_numeric(tmp[target], errors="coerce")
        tmp = tmp.dropna(subset=[age_col, target])
        tmp[target] = tmp[target].astype(int)

        bins = [0, 3, 6, 12, 24, 36, 60, np.inf]
        labels = ["0-3", "4-6", "7-12", "13-24", "25-36", "37-60", "60+"]
        tmp["bin"] = pd.cut(tmp[age_col], bins=bins, labels=labels, include_lowest=True)

        churn_pct = (tmp.groupby("bin")[target].mean() * 100).reindex(labels)
        best_cat, best_val = _safe_top(churn_pct)

        story.append(Paragraph(f"1) Churn por antigüedad ({age_col})", styles["Heading3"]))
        if best_cat:
            story.append(Paragraph(f"Hallazgo: mayor churn en <b>{best_cat}</b> con <b>{best_val:.2f}%</b>.", styles["Normal"]))

        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.bar(churn_pct.index.astype(str), churn_pct.values)
        ax.set_ylabel("Churn (%)")
        ax.set_xlabel("Intervalo")
        story.append(_fig_to_rl_image(fig))
        story.append(Spacer(1, 12))

    # ------------ 2) Churn por Tipo de suscripción ------------
    sub_col = _pick_col(df, ["SubscriptionType", "PlanType", "Plan", "subscription_type", "plan_type"])
    if sub_col:
        tmp2 = df[[sub_col, target]].copy().dropna(subset=[sub_col, target])
        tmp2[target] = pd.to_numeric(tmp2[target], errors="coerce").astype(int)

        churn_by = (tmp2.groupby(sub_col)[target].mean() * 100).sort_values(ascending=False)
        best_cat, best_val = _safe_top(churn_by)

        story.append(Paragraph(f"2) Churn por tipo de suscripción ({sub_col})", styles["Heading3"]))
        if best_cat:
            story.append(Paragraph(f"Hallazgo: mayor churn en <b>{best_cat}</b> con <b>{best_val:.2f}%</b>.", styles["Normal"]))

        # top 8 para que el gráfico no sea infinito
        churn_by_plot = churn_by.head(8).sort_values()
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.barh(churn_by_plot.index.astype(str), churn_by_plot.values)
        ax.set_xlabel("Churn (%)")
        story.append(_fig_to_rl_image(fig))
        story.append(Spacer(1, 12))

    # ------------ 3) Churn por Tickets ------------
    tix_col = _pick_col(df, ["SupportTicketsPerMonth", "SupportTickets", "Tickets", "NumTickets"])
    if tix_col:
        tmp3 = df[[tix_col, target]].copy()
        tmp3[tix_col] = pd.to_numeric(tmp3[tix_col], errors="coerce")
        tmp3[target] = pd.to_numeric(tmp3[target], errors="coerce")
        tmp3 = tmp3.dropna(subset=[tix_col, target])
        tmp3[target] = tmp3[target].astype(int)

        bins_t = [-0.1, 0, 1, 2, 3, 5, np.inf]
        labels_t = ["0", "1", "2", "3", "4-5", "6+"]
        tmp3["bin"] = pd.cut(tmp3[tix_col], bins=bins_t, labels=labels_t, include_lowest=True)

        churn_t = (tmp3.groupby("bin")[target].mean() * 100).reindex(labels_t)
        best_cat, best_val = _safe_top(churn_t)

        story.append(Paragraph(f"3) Churn por tickets ({tix_col})", styles["Heading3"]))
        if best_cat:
            story.append(Paragraph(f"Hallazgo: mayor churn en <b>{best_cat}</b> tickets con <b>{best_val:.2f}%</b>.", styles["Normal"]))

        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.bar(churn_t.index.astype(str), churn_t.values)
        ax.set_ylabel("Churn (%)")
        ax.set_xlabel("Tickets (intervalo)")
        story.append(_fig_to_rl_image(fig))
        story.append(Spacer(1, 12))

    # ------------ 4) Distribución del riesgo (modelo) ------------
    if "proba_churn" in scored.columns:
        story.append(Paragraph("4) Distribución del riesgo (probabilidad de churn)", styles["Heading3"]))

        p50 = float(scored["proba_churn"].median())
        p90 = float(scored["proba_churn"].quantile(0.90))
        story.append(Paragraph(f"Hallazgo: mediana de riesgo <b>{p50:.3f}</b> | P90 <b>{p90:.3f}</b>.", styles["Normal"]))

        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.hist(scored["proba_churn"].dropna().values, bins=30)
        ax.set_xlabel("Probabilidad churn")
        ax.set_ylabel("Clientes")
        story.append(_fig_to_rl_image(fig))
        story.append(Spacer(1, 12))

    # ------------ 5) Ojiva churn acumulado vs TotalCharges ------------
    tc_col = _pick_col(df, ["TotalCharges"])
    if tc_col:
        tmp4 = df[[tc_col, target]].copy()
        tmp4[tc_col] = pd.to_numeric(tmp4[tc_col], errors="coerce")
        tmp4[target] = pd.to_numeric(tmp4[target], errors="coerce")
        tmp4 = tmp4.dropna(subset=[tc_col, target])
        tmp4[target] = tmp4[target].astype(int)

        churners = tmp4[tmp4[target] == 1].sort_values(tc_col)
        if len(churners) > 0:
            # puntos por cuantiles para curva
            qs = np.linspace(0.05, 0.95, 12)
            xs = churners[tc_col].quantile(qs).values
            ys = qs * 100  # % churners acumulado

            story.append(Paragraph(f"5) Ojiva: churn acumulado vs {tc_col}", styles["Heading3"]))
            story.append(Paragraph(
                f"Hallazgo: ~50% del churn ocurre por debajo de <b>{np.quantile(churners[tc_col], 0.50):.0f}</b> y ~80% por debajo de <b>{np.quantile(churners[tc_col], 0.80):.0f}</b>.",
                styles["Normal"]
            ))

            fig, ax = plt.subplots(figsize=(7.2, 2.6))
            ax.plot(xs, ys, marker="o")
            ax.set_xlabel(f"{tc_col} (umbral)")
            ax.set_ylabel("% churn acumulado (dentro de churners)")
            ax.set_ylim(0, 100)
            story.append(_fig_to_rl_image(fig))
            story.append(Spacer(1, 12))

    story.append(Paragraph("Notas", styles["Heading2"]))
    story.append(Paragraph(
        "Los hallazgos muestran patrones de churn por segmentos y la distribución del riesgo estimado por el modelo. "
        "Para intervención, el sistema prioriza clientes del Top 10% de riesgo en el módulo de detalle por cliente.",
        styles["Normal"]
    ))

    doc.build(story)
    return buf.getvalue()


def build_cliente_report_pdf(
    cliente_id: str,
    fila_score: pd.Series,
    cliente_row: pd.Series,
    shap_top5: list,
    recomendaciones: list,
    target: str
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Informe por cliente — Churn (Detalle)", styles["Title"]))
    story.append(Paragraph(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"CustomerID: <b>{cliente_id}</b>", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Resumen del scoring
    prob = float(fila_score.get("proba_churn", np.nan))
    pred = int(fila_score.get("pred_churn", -1))
    top10 = int(fila_score.get("top10_flag", 0))
    rank = fila_score.get("rank", "N/A")

    resumen = [
        ["Campo", "Valor"],
        ["Probabilidad churn", _fmt(prob, 4)],
        ["Predicción (umbral)", str(pred)],
        ["Top 10% riesgo", "Sí" if top10 == 1 else "No"],
        ["Ranking riesgo", str(rank)],
    ]
    t = Table(resumen, colWidths=[220, 300])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Paragraph("Resumen del modelo", styles["Heading2"]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Perfil (solo algunas variables típicas para no llenar el PDF)
    story.append(Paragraph("Perfil (variables clave)", styles["Heading2"]))
    preferidas = [
        "SubscriptionType", "PlanType", "Tenure", "AccountAge", "Tickets", "SupportTicketsPerMonth",
        "ViewingHoursPerWeek", "AverageViewingDuration", "ContentDownloadsPerMonth", "ContentDowloadsPerMonth",
        "MonthlyCharges", "TotalCharges"
    ]
    filas = []
    for c in preferidas:
        if c in cliente_row.index and c != target:
            filas.append([c, str(cliente_row[c])])

    if not filas:
        story.append(Paragraph("No hay variables clave disponibles (o el dataset no contiene esas columnas).", styles["Normal"]))
    else:
        perfil_tbl = Table([["Variable", "Valor"]] + filas, colWidths=[220, 300])
        perfil_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(perfil_tbl)

    story.append(Spacer(1, 12))

    # SHAP top 5 (texto)
    story.append(Paragraph("Top 5 variables de impacto (SHAP — cliente)", styles["Heading2"]))
    if shap_top5:
        rows = [["Variable", "SHAP (impacto)"]]
        for name, val in shap_top5:
            rows.append([str(name), _fmt(float(val), 4)])
        shap_tbl = Table(rows, colWidths=[320, 200])
        shap_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(shap_tbl)
    else:
        story.append(Paragraph("No disponible (SHAP no calculado).", styles["Normal"]))

    story.append(Spacer(1, 12))

    # Recomendaciones (solo si Top10)
    story.append(Paragraph("Intervención recomendada", styles["Heading2"]))
    if top10 != 1:
        story.append(Paragraph("El cliente no está en el Top 10% de riesgo. No se generan recomendaciones.", styles["Normal"]))
    else:
        if not recomendaciones:
            story.append(Paragraph("Top 10% detectado, pero no se activaron reglas con las columnas disponibles.", styles["Normal"]))
        else:
            # Estilo para wrap
            wrap_style = ParagraphStyle(
                name="wrap",
                fontName="Helvetica",
                fontSize=9,
                leading=11,
                wordWrap="CJK",  # wrap agresivo (funciona bien para textos largos)
            )

            head_style = ParagraphStyle(
                name="head",
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=11,
            )

            rows = [
                [
                    Paragraph("Prioridad", head_style),
                    Paragraph("Motivo", head_style),
                    Paragraph("Acción", head_style),
                    Paragraph("Canal", head_style),
                ]
            ]

            for r in recomendaciones:
                rows.append([
                    Paragraph(str(r.get("prioridad", "")), wrap_style),
                    Paragraph(str(r.get("motivo", "")), wrap_style),
                    Paragraph(str(r.get("accion", "")), wrap_style),
                    Paragraph(str(r.get("canal", "")), wrap_style),
                ])

            # Anchos más balanceados
            rec_tbl = Table(rows, colWidths=[55, 140, 255, 90], repeatRows=1)

            rec_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(rec_tbl)

    doc.build(story)
    return buf.getvalue()
