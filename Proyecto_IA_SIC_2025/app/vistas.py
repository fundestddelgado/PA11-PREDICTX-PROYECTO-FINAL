# app/vistas.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

from sklearn.model_selection import train_test_split


def preparar_train_test(df_clean: pd.DataFrame):
    X = df_clean.drop("Exited", axis=1)
    y = df_clean["Exited"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def kpi_card(titulo: str, valor: str, color: str = "#3b82f6"):
    barra = f"linear-gradient(90deg, {color}33, {color})"
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value">{valor}</div>
        <div class="kpi-bottom" style="background: {barra};"></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def apply_dark_theme(fig):
    bg = "#001f3f"
    spine_color = "#4b5563"
    grid_color = "#334155"

    fig.patch.set_facecolor(bg)
    for ax in fig.axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(color=grid_color, alpha=0.4)


def render_resumen(df_raw: pd.DataFrame, df_clean: pd.DataFrame, modelo):
    total_clientes = len(df_clean)
    total_churn = int(df_clean["Exited"].sum())
    tasa_churn = total_churn / total_clientes if total_clientes else 0.0

    # ---- KPIs ----
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Total de clientes", f"{total_clientes}", color="#3b82f6")
    with col2:
        kpi_card("Clientes que se han ido", f"{total_churn}", color="#ef4444")
    with col3:
        kpi_card("Tasa de pérdida de clientes", f"{tasa_churn * 100:.2f}%", color="#10b981")

    # ---- Cálculo clientes de mayor riesgo (7) ----
    df_top_show = None
    try:
        X_all = df_clean.drop("Exited", axis=1)
        proba_all = modelo.predict_proba(X_all)[:, 1]

        df_riesgo = df_raw.copy()
        df_riesgo["RiesgoChurn"] = proba_all

        df_top = df_riesgo.sort_values("RiesgoChurn", ascending=False).head(7)
        df_top_show = df_top[["CustomerId", "Surname", "RiesgoChurn", "Balance"]].copy()
        df_top_show["RiesgoChurn"] = (df_top_show["RiesgoChurn"] * 100).round(2)
        df_top_show.rename(
            columns={
                "CustomerId": "ID Cliente",
                "Surname": "Apellido",
                "RiesgoChurn": "% Riesgo churn",
                "Balance": "Balance",
            },
            inplace=True,
        )
    except Exception as e:
        st.error("No se pudo calcular la lista de clientes con mayor riesgo.")
        st.text(f"Detalle técnico: {e}")

    # contenedor izquierdo (segmentación) y derecho (clientes en mayor riesgo)
    left_box, right_box = st.columns([2, 1], gap="medium")

    # -------- SEGMENTACIÓN --------
    with left_box:
        with st.container(border=True):
            st.markdown("**Segmentación de clientes**")

            seg_col1, seg_col2 = st.columns(2)

            # Churn por grupo de edad
            with seg_col1:
                try:
                    bins = [0, 30, 40, 50, 100]
                    labels = ["0-30", "31-40", "41-50", "51+"]
                    df_age = df_clean.copy()
                    df_age["GrupoEdad"] = pd.cut(df_age["Age"], bins=bins, labels=labels, include_lowest=True)

                    churn_por_edad = df_age.groupby("GrupoEdad", observed=True)["Exited"].mean().reset_index()
                    churn_por_edad["tasa_churn_pct"] = churn_por_edad["Exited"] * 100

                    fig_e, ax_e = plt.subplots(figsize=(2.4, 1.6))
                    sns.barplot(data=churn_por_edad, x="GrupoEdad", y="tasa_churn_pct", ax=ax_e)

                    ax_e.set_ylabel("Churn (%)", fontsize=7)
                    ax_e.set_xlabel("Grupo de edad", fontsize=7)
                    ax_e.tick_params(axis="x", labelsize=6)
                    ax_e.tick_params(axis="y", labelsize=6)

                    apply_dark_theme(fig_e)
                    st.pyplot(fig_e)
                except Exception:
                    st.info("No se pudo graficar la segmentación por edad.")

            # Churn por IsActiveMember
            with seg_col2:
                try:
                    churn_por_actividad = df_clean.groupby("IsActiveMember")["Exited"].mean().reset_index()
                    churn_por_actividad["EstadoActividad"] = churn_por_actividad["IsActiveMember"].map(
                        {0: "No activo", 1: "Activo"}
                    )
                    churn_por_actividad["tasa_churn_pct"] = churn_por_actividad["Exited"] * 100

                    fig_a, ax_a = plt.subplots(figsize=(2.4, 1.6))
                    sns.barplot(data=churn_por_actividad, x="EstadoActividad", y="tasa_churn_pct", ax=ax_a)

                    ax_a.set_ylabel("Churn (%)", fontsize=7)
                    ax_a.set_xlabel("Nivel de actividad", fontsize=7)
                    ax_a.tick_params(axis="x", labelsize=6)
                    ax_a.tick_params(axis="y", labelsize=6)

                    apply_dark_theme(fig_a)
                    st.pyplot(fig_a)
                except Exception:
                    st.info("No se pudo graficar la segmentación por actividad.")

    # --------  CLIENTES CON MAYOR RIESGO --------
    with right_box:
        with st.container(border=True):
            st.markdown("**Clientes con mayor riesgo**")
            if df_top_show is not None:
                st.dataframe(df_top_show, hide_index=True, height=300)
            else:
                st.write("No se pudo cargar la tabla de clientes.")

    # -------- SHAP GLOBAL --------
    try:
        pre = modelo.named_steps["preprocesamiento"]
        clf = modelo.named_steps["clasificador"]

        num_features = list(pre.transformers_[0][2])
        ohe = pre.transformers_[1][1]
        cat_orig = pre.transformers_[1][2]
        cat_features = list(ohe.get_feature_names_out(cat_orig))
        feature_names = num_features + cat_features

        X_train, X_test, y_train, y_test = preparar_train_test(df_clean)
        X_train_t = pre.transform(X_train)
        if hasattr(X_train_t, "toarray"):
            X_train_t = X_train_t.toarray()

        n_muestra = min(1000, X_train_t.shape[0])
        idx_sample = np.random.choice(X_train_t.shape[0], n_muestra, replace=False)
        X_sample = X_train_t[idx_sample, :]
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

        explainer_global = shap.TreeExplainer(clf)
        shap_values = explainer_global.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            if shap_values.ndim == 3 and shap_values.shape[2] >= 2:
                shap_values_class1 = shap_values[:, :, 1]
            else:
                shap_values_class1 = shap_values

        shap_col1, shap_col2 = st.columns(2, gap="medium")

        with shap_col1:
            with st.container(border=True):
                st.markdown("**Importancia media**")
                fig_shap_bar = plt.figure(figsize=(3, 2))
                shap.summary_plot(
                    shap_values_class1,
                    X_sample_df,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False,
                )
                apply_dark_theme(fig_shap_bar)
                st.pyplot(fig_shap_bar, clear_figure=True)

        with shap_col2:
            with st.container(border=True):
                st.markdown("**Impacto de las características de los clientes**")
                fig_shap_bee = plt.figure(figsize=(3, 2))
                shap.summary_plot(
                    shap_values_class1,
                    X_sample_df,
                    feature_names=feature_names,
                    show=False,
                )
                apply_dark_theme(fig_shap_bee)
                st.pyplot(fig_shap_bee, clear_figure=True)

    except Exception as e:
        st.error("No se pudo calcular la explicabilidad global con SHAP.")
        st.text(f"Detalle técnico: {e}")


def render_cliente(df_raw: pd.DataFrame, df_clean: pd.DataFrame, modelo):
    df_with_id = pd.concat([df_raw[["CustomerId"]], df_clean], axis=1)

    # ---- Selección de cliente por ID ----
    id_options = df_with_id["CustomerId"].tolist()
    sel_id = st.selectbox("Selecciona un CustomerId", id_options)

    fila_cliente = df_with_id[df_with_id["CustomerId"] == sel_id].iloc[0]
    features_cliente = fila_cliente.drop(labels=["CustomerId", "Exited"])

    features_df = pd.DataFrame([features_cliente])

    fila_raw = df_raw[df_raw["CustomerId"] == sel_id].iloc[0]
    surname = fila_raw["Surname"] if "Surname" in fila_raw.index else ""

    # ---- Predicción del modelo para este cliente ----
    proba_churn = modelo.predict_proba(features_df)[0, 1]
    pred_label = modelo.predict(features_df)[0]

    if proba_churn < 0.33:
        nivel_riesgo = "Bajo"
    elif proba_churn < 0.66:
        nivel_riesgo = "Medio"
    else:
        nivel_riesgo = "Alto"

    # ---- KPIs  ----
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        kpi_card("Probabilidad de abandono", f"{proba_churn * 100:.2f}%", color="#ef4444")

    with col_m2:
        kpi_card(
            "Clasificación según el modelo",
            "En riesgo de irse" if pred_label == 1 else "Poca posibilidad",
            color="#3b82f6",
        )

    color_riesgo = {"Bajo": "#10b981", "Medio": "#f59e0b", "Alto": "#ef4444"}.get(nivel_riesgo, "#10b981")
    with col_m3:
        kpi_card("Nivel de riesgo", nivel_riesgo, color=color_riesgo)

    # ----  info / SHAP local ----
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        with st.container(border=True):
            st.markdown("**Información del cliente**")

            info_df = features_cliente.to_frame(name="Valor")
            info_df.index.name = "Campo"

            surname_df = pd.DataFrame({"Campo": ["Apellido"], "Valor": [surname]}).set_index("Campo")
            info_df = pd.concat([surname_df, info_df])

            st.dataframe(info_df, use_container_width=True, height=380)

    with col_right:
        with st.container(border=True):
            st.markdown("**Explicación local (Waterfall)**")

            try:
                pre = modelo.named_steps["preprocesamiento"]
                clf = modelo.named_steps["clasificador"]

                num_features = list(pre.transformers_[0][2])
                ohe = pre.transformers_[1][1]
                cat_orig = pre.transformers_[1][2]
                cat_features = list(ohe.get_feature_names_out(cat_orig))
                feature_names = num_features + cat_features

                X_cliente_t = pre.transform(features_df)
                if hasattr(X_cliente_t, "toarray"):
                    X_cliente_t = X_cliente_t.toarray()
                x_client = X_cliente_t[0]

                explainer_local = shap.TreeExplainer(clf)
                shap_vals_local = explainer_local.shap_values(X_cliente_t)

                if isinstance(shap_vals_local, list):
                    shap_client_class1 = shap_vals_local[1][0]
                    base_value = explainer_local.expected_value[1]
                else:
                    if shap_vals_local.ndim == 3 and shap_vals_local.shape[2] >= 2:
                        shap_client_class1 = shap_vals_local[0, :, 1]
                        base_value = explainer_local.expected_value[1]
                    else:
                        shap_client_class1 = shap_vals_local[0]
                        base_value = explainer_local.expected_value

                explanation = shap.Explanation(
                    values=shap_client_class1,
                    base_values=base_value,
                    data=x_client,
                    feature_names=feature_names,
                )

                fig_wf = plt.figure(figsize=(8, 5))
                shap.plots.waterfall(explanation, max_display=10, show=False)
                apply_dark_theme(fig_wf)
                st.pyplot(fig_wf, clear_figure=True)

            except Exception as e:
                st.error("No se pudo generar la explicación local para este cliente.")
                st.text(f"Detalle técnico: {e}")
