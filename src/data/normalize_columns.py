import pandas as pd
import re

# Diccionario de nombres estandarizados (en español)
MAPEO_COLUMNAS = {
    "jugador": "jugador",
    "pareja": "pareja",
    "punto_win": "punto_ganado",
    "punto_lost": "punto_perdido",
    "winner": "winner",
    "error": "error_no_forzado",
    "fuerza_error": "error_forzado",
    "set_num": "set_num",
    "juego_p1": "juego_p1",
    "juego_p2": "juego_p2",
    # Golpes con errores ortográficos incluidos
    "inicio_golpe": "golpe_inicio",
    "inicio_gople": "golpe_inicio",
    "fin_golpe": "golpe_fin",
    "zona_saque": "zona_saque",
    "zona_resto": "zona_resto",
    "golpe_q": "golpe_q"
}

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Renombrar según diccionario
    df = df.rename(columns={c: MAPEO_COLUMNAS.get(c, c) for c in df.columns})

    # 2) Fusionar columnas _time / _x / _y
    patron = r"(.+):_(time|x|y)$"
    agrupadas = {}

    for col in df.columns:
        match = re.match(patron, col)
        if match:
            base, suf = match.groups()
            agrupadas.setdefault(base, {})[suf] = col

    for base, partes in agrupadas.items():
        if "time" in partes:
            df[f"{base}_t"] = df.get(f"{base}_t") or df[partes["time"]]
        if "x" in partes:
            df[f"{base}_x"] = df.get(f"{base}_x") or df[partes["x"]]
        if "y" in partes:
            df[f"{base}_y"] = df.get(f"{base}_y") or df[partes["y"]]

        df = df.drop(columns=list(partes.values()), errors="ignore")

    # 3) Quitar columnas totalmente vacías
    df = df.dropna(axis=1, how="all")

    # 4) Eliminar duplicadas por nombre
    df = df.loc[:, ~df.columns.duplicated()]

    return df
