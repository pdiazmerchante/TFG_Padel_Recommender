"""
Limpieza completa del dataset M3 de pádel.
------------------------------------------
Este módulo transforma los datos crudos tras el colapso de eventos en un
formato limpio y homogéneo para análisis. Incluye:

1️⃣ Renombrado de columnas a nombres estándar.
2️⃣ Homogeneización de texto (minúsculas, sin espacios ni tildes).
3️⃣ Normalización de categorías (winner, error_nf, error_f, etc.).
4️⃣ Conversión de columnas numéricas (manejo de comas, espacios).
5️⃣ Eliminación de duplicados y valores nulos textuales.
6️⃣ Conversión final de tipos (string, float, Int64).
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# === 1️⃣ MAPEOS DE COLUMNAS ORIGINALES → NUEVOS NOMBRES ===
# Esto permite que, sin importar cómo venga el CSV (en inglés, español o abreviado),
# siempre obtengas nombres uniformes en snake_case.
COLUMN_MAP = {
    # Identificación
    "player": "jugador",
    "nombre": "jugador",
    "pair": "pareja",
    "rivales": "rival",
    "rival": "rival",
    "point_id": "punto_id",
    "id_punto": "punto_id",
    "punto": "punto_id",

    # Golpe / resultado
    "shot": "golpe",
    "stroke": "golpe",
    "winner": "es_winner",
    "resultado_golpe": "resultado",
    "forced_error": "error_f",
    "unforced_error": "error_nf",
    "error_no_forzado": "error_nf",
    "error_forzado": "error_f",

    # Coordenadas / posición
    "start_shot_x": "inicio_golpe_x",
    "start_shot_y": "inicio_golpe_y",
    "pos_x": "inicio_golpe_x",
    "pos_y": "inicio_golpe_y",

    # Otros posibles alias
    "serve": "saque",
    "side": "lado",
}

# === 2️⃣ COLUMNAS DE TIPO STRING Y NUMÉRICO ===
STR_COLS_LIKE = [
    "jugador", "pareja", "rival", "golpe", "resultado",
    "lado", "torneo"
]
NUM_COLS_LIKE = [
    "punto_id", "set", "game", "inicio_golpe_x",
    "inicio_golpe_y", "velocidad", "prob_xg"
]

# === 3️⃣ NORMALIZACIÓN DE CATEGORÍAS ===
# Para convertir todas las etiquetas distintas a una única forma coherente.
CATEG_NORMALIZE = {
    "resultado": {
        # Ganadores
        "winner": "winner", "w": "winner", "ganador": "winner",
        # Errores no forzados
        "error no forzado": "error_nf", "enf": "error_nf",
        "error_nf": "error_nf",
        # Errores forzados
        "error forzado": "error_f", "ef": "error_f",
        "error_f": "error_f"
    },
    "golpe": {
        "drive": "drive",
        "revés": "reves", "reves": "reves",
        "bandeja": "bandeja",
        "volea": "volea",
        "smash": "smash",
        "globo": "globo",
        "dejada": "dejada",
        "remate": "smash",
    },
    "lado": {
        "derecha": "derecha",
        "izquierda": "izquierda",
        "revés": "reves",
        "reves": "reves",
    }
}

# === FUNCIONES AUXILIARES ===

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas según el mapeo y aplica snake_case y sin tildes."""
    rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    def to_snake(c: str) -> str:
        return (
            c.strip()
             .lower()
             .replace(" ", "_")
             .replace("-", "_")
             .replace("ó", "o").replace("á", "a")
             .replace("é", "e").replace("í", "i")
             .replace("ú", "u").replace("ñ", "n")
        )
    df.columns = [to_snake(c) for c in df.columns]
    return df


def normalize_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Quita espacios, pasa a minúsculas y homogeneiza texto."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.lower()
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convierte columnas numéricas a float, gestionando comas decimales."""
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("string")
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# === FUNCIÓN PRINCIPAL ===

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza completa del dataset de pádel M3."""
    df = df.copy()

    # 1️⃣ Renombrar columnas y pasarlas a snake_case
    df = standardize_columns(df)

    # 2️⃣ Reemplazar valores vacíos / falsos nulos
    # df = df.replace(["", "NA", "na", "null", "None"], np.nan)

    # 3️⃣ Eliminar duplicados exactos
    df = df.drop_duplicates()

    # 4️⃣ Eliminar columnas que contengan ':time' en el nombre
    cols_a_eliminar = [c for c in df.columns if ":time" in c.lower()]
    if cols_a_eliminar:
        print(f"🧹 Eliminando columnas con ':time': {cols_a_eliminar}")
        df = df.drop(columns=cols_a_eliminar)

    # 5️⃣ Ajustar tipos de datos modernos (pandas 2.x)
    df = df.convert_dtypes()

    # ==========================================
    # 6️⃣ NORMALIZACIÓN FINAL DE COORDENADAS
    # ==========================================
    COORD_MAP = {
        # posibles variantes originales
        "inicio_golpe_x": "inicio_x",
        "inicio_gople_x": "inicio_x",
        "inicio_golpe:y": "inicio_y",
        "inicio_gople:y": "inicio_y",
        "inicio_golpe:x": "inicio_x",
        "inicio_golpe:y": "inicio_y",
        "inicio_golpe_x": "inicio_x",
        "inicio_golpe_y": "inicio_y",

        # tu CSV original
        "inicio_gople: x": "inicio_x",
        "inicio_gople: y": "inicio_y",
        "fin_golpe: x": "fin_x",
        "fin_golpe: y": "fin_y",

        # versiones de normalizar_columnas
        "inicio_golpe_x": "inicio_x",
        "inicio_golpe_y": "inicio_y",
        "golpe_inicio_x": "inicio_x",
        "golpe_inicio_y": "inicio_y",
        "golpe_fin_x": "fin_x",
        "golpe_fin_y": "fin_y",
    }

    df = df.rename(columns={c: COORD_MAP[c] for c in df.columns if c in COORD_MAP})

    # asegurar que existan aunque estén vacías (para no borrarlas después)
    for col in ["inicio_x", "inicio_y", "fin_x", "fin_y"]:
        if col not in df.columns:
            df[col] = pd.NA
    # 7️⃣ NO ELIMINAR columnas de coordenadas aunque estén vacías
    cols_empty = [
        c for c in df.columns 
        if df[c].isna().all() and c not in ["inicio_x", "inicio_y", "fin_x", "fin_y"]
    ]
    df = df.drop(columns=cols_empty, errors="ignore")


    return df

