# nucleo/carga_y_datos.py
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def cargar_csv(ruta: Union[str, Path]) -> pd.DataFrame:

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {ruta}")
    return pd.read_csv(ruta)


def limpiar_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    # Columnas que no aportan al modelo
    columnas_a_eliminar = ["RowNumber", "CustomerId", "Surname"]

    existentes = [c for c in columnas_a_eliminar if c in df.columns]
    if existentes:
        df = df.drop(columns=existentes)

    return df


@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta: Union[str, Path]):

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo del modelo: {ruta}")
    return joblib.load(ruta)
