import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def render_general(df: pd.DataFrame, scored: pd.DataFrame, fuente: str, target: str):
    st.subheader("Resumen general")

    c1, c2 = st.columns(2)
    c1.metric("Clientes totales", f"{len(df):,}")

    if target in df.columns:
        churn_rate = float(df[target].mean())
        c2.metric("Porcentaje de churn", f"{100 * churn_rate:.2f}%")
    else:
        c2.metric("Porcentaje de churn", "N/A")
        st.warning(f"El dataset no trae la columna '{target}', no se puede calcular churn rate.")

    st.subheader("Churn por segmentos")
    colA, colB, colC = st.columns(3)

    # 1) Antigüedad
    with colA:
        st.caption("Antigüedad (intervalos)")
        age_col = next((c for c in ["AccountAge", "account_age", "Tenure", "tenure", "EdadCuenta", "edad_cuenta"] if c in df.columns), None)

        if (age_col is None) or (target not in df.columns):
            st.info("Falta AccountAge/Tenure y/o Churn.")
        else:
            tmp = df[[age_col, target]].copy()
            tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
            tmp = tmp.dropna(subset=[age_col, target])

            bins = [0, 3, 6, 12, 24, 36, 60, np.inf]
            labels = ["0-3", "4-6", "7-12", "13-24", "25-36", "37-60", "60+"]

            tmp["bin"] = pd.cut(tmp[age_col], bins=bins, labels=labels, include_lowest=True)
            churn_pct = (tmp.groupby("bin")[target].mean() * 100).reindex(labels)

            fig, ax = plt.subplots(figsize=(3.6, 2.6))
            ax.bar(churn_pct.index.astype(str), churn_pct.values)
            ax.set_ylabel("Churn (%)")
            ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            st.pyplot(fig, use_container_width=True)

    # 2) Tipo de suscripción
    with colB:
        st.caption("Tipo de suscripción")
        sub_col = next((c for c in ["SubscriptionType", "subscription_type", "Plan", "PlanType", "plan_type", "TipoSuscripcion", "tipo_suscripcion"] if c in df.columns), None)

        if (sub_col is None) or (target not in df.columns):
            st.info("Falta SubscriptionType (o similar) y/o Churn.")
        else:
            tmp2 = df[[sub_col, target]].copy().dropna(subset=[sub_col, target])
            churn_by = (tmp2.groupby(sub_col)[target].mean() * 100).sort_values(ascending=False)

            fig2, ax2 = plt.subplots(figsize=(3.6, 2.6))
            ax2.bar(churn_by.index.astype(str), churn_by.values)
            ax2.set_ylabel("Churn (%)")
            ax2.tick_params(axis="x", rotation=35, labelsize=8)
            ax2.tick_params(axis="y", labelsize=8)
            st.pyplot(fig2, use_container_width=True)

    # 3) Tickets
    with colC:
        st.caption("Tickets de soporte (intervalos)")
        tix_col = next((c for c in ["SupportTicketsPerMonth", "SupportTickets", "Tickets", "tickets", "NumTickets", "num_tickets"] if c in df.columns), None)

        if (tix_col is None) or (target not in df.columns):
            st.info("Falta columna de tickets y/o Churn.")
        else:
            tmp3 = df[[tix_col, target]].copy()
            tmp3[tix_col] = pd.to_numeric(tmp3[tix_col], errors="coerce")
            tmp3 = tmp3.dropna(subset=[tix_col, target])

            bins_t = [-0.1, 0, 1, 2, 3, 5, np.inf]
            labels_t = ["0", "1", "2", "3", "4-5", "6+"]

            tmp3["bin"] = pd.cut(tmp3[tix_col], bins=bins_t, labels=labels_t, include_lowest=True)
            churn_t = (tmp3.groupby("bin")[target].mean() * 100).reindex(labels_t)

            fig3, ax3 = plt.subplots(figsize=(3.6, 2.6))
            ax3.bar(churn_t.index.astype(str), churn_t.values)
            ax3.set_ylabel("Churn (%)")
            ax3.tick_params(axis="x", rotation=0, labelsize=8)
            ax3.tick_params(axis="y", labelsize=8)
            st.pyplot(fig3, use_container_width=True)

    st.subheader("Vista rápida del dataset")
    st.caption(f"Fuente: {fuente}")
    st.dataframe(df.head(10), use_container_width=True, height=260)
