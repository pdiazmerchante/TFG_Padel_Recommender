import pandas as pd
import numpy as np

# ============================================================
# 1) Inferir parejas
# ============================================================
def inferir_parejas(df):
    df = df.copy()
    parejas = df["pareja"].dropna().unique()

    if len(parejas) < 2:
        raise ValueError("No se detectaron dos parejas distintas.")

    pareja1 = parejas[0]
    pareja2 = [p for p in parejas if p != pareja1][0]

    df["pareja_id"] = df["pareja"].apply(lambda x: 1 if x == pareja1 else 2)
    return df, pareja1, pareja2


# ============================================================
# 2) Extraer sacador del texto "1º Servicio Nombre Apellido"
# ============================================================
def extraer_sacador(df):
    df = df.copy()

    def get_sacador(text):
        if pd.isna(text):
            return None
        parts = str(text).strip().split(" ")
        if len(parts) < 3:
            return None
        return " ".join(parts[2:]).strip()

    df["saca_jugador"] = df["servicio"].apply(get_sacador)

    # Identificar pareja que saca
    df["saca_pareja"] = df.apply(
        lambda r: r["pareja_id"] if r["jugador"] == r["saca_jugador"] else None,
        axis=1
    )

    return df


# ============================================================
# 3) Inferir quién gana cada punto comparando marcador anterior
# ============================================================
def inferir_ganador_punto(df):
    df = df.copy()

    df["punto_p1_prev"] = df["punto_p1"].shift(1)
    df["punto_p2_prev"] = df["punto_p2"].shift(1)

    def ganador(row):
        if row["punto_p1"] != row["punto_p1_prev"]:
            return 1
        if row["punto_p2"] != row["punto_p2_prev"]:
            return 2
        return None

    df["ganador_pareja"] = df.apply(ganador, axis=1)

    return df


# ============================================================
# 4) Etiqueta: ¿ganó el punto el sacador?
# ============================================================
def etiquetar_puntos_saque(df):
    df = df.copy()
    df["gana_punto_sacador"] = (df["ganador_pareja"] == df["saca_pareja"])
    return df


# ============================================================
# 5) NUEVA FUNCIÓN CORRECTA: estadísticas por pareja
# ============================================================
def resumen_estadisticas_saque(df, pareja1, pareja2):
    """
    Calcula estadísticas de saque por cada pareja.
    La firma ACEPTA 3 ARGUMENTOS: (df, pareja1, pareja2)
    """

    df = df.copy()
    datos = []

    for pareja_id, nombre in [(1, pareja1), (2, pareja2)]:

        puntos_saque = df[df["saca_pareja"] == pareja_id]
        total_puntos = len(puntos_saque)
        puntos_ganados = puntos_saque["gana_punto_sacador"].sum()
        pct = (puntos_ganados / total_puntos * 100) if total_puntos > 0 else 0

        # Juegos: se gana un juego si algún punto de saque del juego = True
        juegos_saque = df[df["saca_pareja"] == pareja_id]
        juegos_ganados = juegos_saque.groupby("juego_real")["gana_punto_sacador"].max().sum()

        # Breaks sufridos = juegos donde no ganó ningún punto al saque
        breaks_sufridos = juegos_saque.groupby("juego_real")["gana_punto_sacador"].max().apply(lambda x: x is False).sum()

        datos.append({
            "pareja": nombre,
            "puntos_saque_ganados": puntos_ganados,
            "puntos_saque_totales": total_puntos,
            "puntos_saque_%": round(pct, 2),
            "juegos_ganados_al_saque": juegos_ganados,
            "breaks_sufridos": breaks_sufridos
        })

    return pd.DataFrame(datos)
