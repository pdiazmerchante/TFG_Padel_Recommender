import pandas as pd
import numpy as np


# ============================================================
# 1) INFERIR PAREJAS
# ============================================================
def inferir_parejas(df):
    """
    Determina pareja 1 y pareja 2 según el primer valor que aparezca.
    Crea columna pareja_id = 1 ó 2.
    """
    df = df.copy()

    parejas = [p for p in df["pareja"].dropna().unique() if p != ""]

    if len(parejas) < 2:
        raise ValueError("No se detectaron dos parejas distintas en columna 'pareja'.")

    pareja1 = parejas[0]
    pareja2 = parejas[1]

    df["pareja_id"] = df["pareja"].apply(lambda x: 1 if x == pareja1 else 2)

    return df, pareja1, pareja2


# ============================================================
# 2) EXTRAER SACADOR
# ============================================================
def extraer_sacador(df):
    """
    Extrae nombre del sacador desde columna 'servicio':
    Ejemplo:  '1º Servicio Federico Chingotto'
    """
    df = df.copy()

    def obtener(serv):
        if pd.isna(serv):
            return None
        parts = str(serv).strip().split(" ")
        if len(parts) < 3:
            return None
        return " ".join(parts[2:]).strip()

    df["saca_jugador"] = df["servicio"].apply(obtener)

    # comparación segura
    df["jugador_clean"] = df["jugador"].astype(str).str.strip()
    df["saca_clean"] = df["saca_jugador"].astype(str).str.strip()

    df["saca_pareja"] = df.apply(
        lambda r: r["pareja_id"] if r["jugador_clean"] == r["saca_clean"] and r["saca_clean"] != "" else None,
        axis=1
    )

    df = df.drop(columns=["jugador_clean", "saca_clean"])
    return df


# ============================================================
# 3) INFERIR GANADOR DEL PUNTO
# ============================================================
def inferir_ganador_punto(df):
    """
    Detecta el ganador del punto comparando cambio en marcador_puntos.
    SOLO se marca ganador cuando cambia 0/15/30/40/AD.
    """
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
# 4) ¿EL SACADOR GANÓ EL PUNTO?
# ============================================================
def etiquetar_puntos_saque(df):
    df = df.copy()

    df["gana_punto_sacador"] = df.apply(
        lambda r: True if r["ganador_pareja"] == r["saca_pareja"] else False
        if (r["ganador_pareja"] is not None and r["saca_pareja"] is not None)
        else None,
        axis=1
    )

    return df


# ============================================================
# 5) ESTADÍSTICAS COMPLETAS DE SAQUE (POR JUGADOR)
# ============================================================
def resumen_estadisticas_saque(df, pareja1, pareja2):
    """
    Devuelve estadísticas POR JUGADOR:
    - puntos ganados al saque
    - puntos totales al saque
    - % puntos al saque
    - juegos ganados al saque
    - juegos perdidos al saque
    """
    df = df.copy()

    datos = []

    # jugadores únicos
    jugadores = df["jugador"].dropna().unique()

    for jugador in jugadores:

        # pareja 1 o 2 de ese jugador
        pareja_id = df.loc[df["jugador"] == jugador, "pareja_id"].iloc[0]

        # puntos donde ese jugador sacó
        puntos_saque = df[df["saca_jugador"] == jugador]

        total = len(puntos_saque)
        ganados = puntos_saque["gana_punto_sacador"].sum() if total > 0 else 0
        pct = (ganados / total * 100) if total > 0 else 0

        # juegos del jugador cuando sacaba su pareja
        juegos_saque = df[df["saca_pareja"] == pareja_id]

        # juego ganado al saque = algún punto ganado por sacador durante ese juego
        juegos_ganados = (
            juegos_saque.groupby("juego_real")["gana_punto_sacador"]
            .max()
            .fillna(False)
            .sum()
        )

        juegos_totales = juegos_saque["juego_real"].nunique()

        juegos_perdidos = juegos_totales - juegos_ganados

        datos.append({
            "jugador": jugador,
            "pareja_id": pareja_id,
            "puntos_saque_ganados": int(ganados),
            "puntos_saque_totales": int(total),
            "puntos_saque_%": round(pct, 2),
            "juegos_saque_ganados": int(juegos_ganados),
            "juegos_saque_perdidos": int(juegos_perdidos),
            "juegos_saque_totales": int(juegos_totales)
        })

    return pd.DataFrame(datos)
