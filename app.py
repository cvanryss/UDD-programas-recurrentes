import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Predictor de Matrícula UDD",
    layout="centered"
)

# =========================================================
# CARGA DE ARCHIVOS
# =========================================================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_programas_recurrentes.pkl")
    programas_master = joblib.load("programas_master.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return modelo, programas_master, feature_cols


modelo, programas_master, feature_cols = cargar_modelo()

# =========================================================
# FUNCIONES
# =========================================================

def score_exito(alumnos):
    bins = [0, 2, 5, 9, 13, 17, 23, 31, 44, 60, 999]

    for i in range(len(bins) - 1):
        if bins[i] <= alumnos < bins[i + 1]:
            return i + 1

    return 10


def clasificar_tendencia(tendencia):
    if tendencia > 5:
        return "Positiva"
    elif tendencia < -5:
        return "Negativa"
    else:
        return "Estable"


def predecir_programa(nombre_programa):

    fila = programas_master[
        programas_master["Nombre Programa"]
        .astype(str)
        .str.lower()
        .str.strip()
        ==
        nombre_programa.lower().strip()
    ]

    if len(fila) == 0:
        return None

    X_pred = fila[feature_cols].copy()

    pred_log = modelo.predict(X_pred)
    alumnos = float(np.expm1(pred_log[0]))

    score = score_exito(alumnos)

    # Error promedio histórico del modelo para programas recurrentes
    margen = 4
    inferior = max(0, round(alumnos - margen))
    superior = round(alumnos + margen)

    ultima_version = round(float(fila["hist_ultima_version"].iloc[0]), 1)
    promedio_historico = round(float(fila["hist_promedio_alumnos"].iloc[0]), 1)
    tendencia = round(float(fila["hist_tendencia_ult"].iloc[0]), 1)
    versiones_previas = int(fila["hist_num_versiones_previas"].iloc[0])

    return {
        "programa": nombre_programa,
        "alumnos_esperados": round(alumnos, 1),
        "score": score,
        "rango": f"{inferior} - {superior}",
        "confianza": "Alta",
        "ultima_version": ultima_version,
        "promedio_historico": promedio_historico,
        "tendencia": tendencia,
        "tendencia_texto": clasificar_tendencia(tendencia),
        "versiones_previas": versiones_previas,
    }


# =========================================================
# INTERFAZ
# =========================================================

st.title("Predictor de Matrícula UDD")

st.caption(
    "Herramienta de apoyo para proyectar la matrícula esperada "
    "de programas recurrentes de Educación Continua."
)

st.info(
    "Este modelo está diseñado para programas con historia previa. "
    "No está pensado para evaluar programas nuevos ni para simular efectos causales "
    "de precio, modalidad u horas."
)

# =========================================================
# SELECTOR DE PROGRAMA
# =========================================================

programas = sorted(
    programas_master["Nombre Programa"]
    .dropna()
    .astype(str)
    .unique()
)

programa = st.selectbox(
    "Selecciona un programa recurrente",
    programas
)

if st.button("Predecir matrícula"):

    resultado = predecir_programa(programa)

    if resultado is None:
        st.error("Programa no encontrado.")
    else:
        st.divider()

        st.subheader("Predicción de matrícula")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Alumnos esperados",
            resultado["alumnos_esperados"]
        )

        col2.metric(
            "Rango probable",
            resultado["rango"]
        )

        col3.metric(
            "Score",
            f'{resultado["score"]}/10'
        )

        st.success(f'Nivel de confianza: {resultado["confianza"]}')

        st.divider()

        st.subheader("Historial del programa")

        col4, col5, col6, col7 = st.columns(4)

        col4.metric(
            "Última versión",
            resultado["ultima_version"]
        )

        col5.metric(
            "Promedio histórico",
            resultado["promedio_historico"]
        )

        col6.metric(
            "Tendencia",
            resultado["tendencia"],
            resultado["tendencia_texto"]
        )

        col7.metric(
            "Versiones previas",
            resultado["versiones_previas"]
        )

        st.caption(
            "El rango probable se construye usando el error promedio observado "
            "en la validación del modelo para programas recurrentes."
        )

# =========================================================
# INFORMACIÓN DEL MODELO
# =========================================================

st.divider()

st.subheader("Información del modelo")

col_a, col_b = st.columns(2)

col_a.metric(
    "R² validación",
    "0.84"
)

col_b.metric(
    "MAE validación",
    "3.75 alumnos"
)

st.write(
    """
**Modelo:** Predictor de Matrícula UDD v1.0  
**Ámbito de uso:** Programas recurrentes con historia previa.  
**Datos utilizados:** Programas históricos de Educación Continua UDD 2022-2025.  

**Interpretación:**  
El modelo explica aproximadamente el 84% de la variación observada en la matrícula
de programas recurrentes y presenta un error promedio cercano a 4 alumnos.

**Limitaciones:**  
- No está diseñado para programas nuevos.
- No debe interpretarse como un simulador causal de precio, modalidad u horas.
- Su principal fortaleza es proyectar nuevas versiones de programas con historial.
"""
)