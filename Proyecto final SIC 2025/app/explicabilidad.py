# app/explicabilidad.py
import numpy as np
import shap
import matplotlib.pyplot as plt


def _partes_pipeline(modelo):

    pre = None
    clf = modelo

    if hasattr(modelo, "named_steps"):
        pre = modelo.named_steps.get("preprocesamiento", None)
        clf = modelo.named_steps.get("clasificador", None)

        if clf is None:
            # último step que prediga
            for _, step in reversed(list(modelo.named_steps.items())):
                if hasattr(step, "predict_proba"):
                    clf = step
                    break

        if pre is None:
            # primer step que transforme
            for _, step in list(modelo.named_steps.items()):
                if hasattr(step, "transform"):
                    pre = step
                    break

    return pre, clf


def _transformar(pre, X):
    return pre.transform(X) if pre is not None else X


def _nombres_features(pre, X_t):
    if pre is not None and hasattr(pre, "get_feature_names_out"):
        try:
            return list(pre.get_feature_names_out())
        except Exception:
            pass

    if hasattr(X_t, "columns"):
        return list(X_t.columns)

    n = X_t.shape[1]
    return [f"f{i}" for i in range(n)]


def _clase_1(shap_values):
    # list[class0, class1]
    if isinstance(shap_values, list) and len(shap_values) >= 2:
        return shap_values[1]
    # ndarray (n, m, 2)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[-1] >= 2:
        return shap_values[:, :, 1]
    # ya viene (n, m)
    return shap_values


def calcular_shap_global(modelo, X, max_muestra=800, random_state=42):

    pre, clf = _partes_pipeline(modelo)

    n = X.shape[0]
    m = min(max_muestra, n)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=m, replace=False)
    X_muestra = X.iloc[idx]

    try:
        X_t = _transformar(pre, X_muestra)
        nombres = _nombres_features(pre, X_t)
        explainer = shap.TreeExplainer(clf)
        shap_vals = _clase_1(explainer.shap_values(X_t))
        return shap_vals, X_t, nombres
    except Exception:
        return None


def figura_shap_barra(shap_vals, X_t, nombres):
    plt.figure()
    shap.summary_plot(shap_vals, X_t, feature_names=nombres, plot_type="bar", show=False)
    plt.title("Importancia global ")
    return plt.gcf()


def figura_shap_beeswarm(shap_vals, X_t, nombres):
    plt.figure()
    shap.summary_plot(shap_vals, X_t, feature_names=nombres, show=False)
    plt.title("Impacto global (beeswarm)")
    return plt.gcf()


def figura_shap_waterfall(modelo, X_una_fila, max_display=10):

    pre, clf = _partes_pipeline(modelo)

    try:
        X_t = _transformar(pre, X_una_fila)
        explainer = shap.TreeExplainer(clf)

        # API nueva: explainer(X) -> Explanation
        try:
            exp = explainer(X_t)[0]
            plt.figure()
            shap.plots.waterfall(exp, max_display=max_display, show=False)
            return plt.gcf()
        except Exception:
            # API antigua fallback
            shap_vals = _clase_1(explainer.shap_values(X_t))
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)) and len(np.atleast_1d(base)) >= 2:
                base = base[1]

            data = np.array(X_t)[0]
            nombres = _nombres_features(pre, X_t)
            explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=base,
                data=data,
                feature_names=nombres
            )
            plt.figure()
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            return plt.gcf()
    except Exception:
        return None

