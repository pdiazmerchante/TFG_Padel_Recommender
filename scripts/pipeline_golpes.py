#!/usr/bin/env python3
# ================================================================
# PIPELINE COMPLETO — reconstrucción marcador + análisis + gráficos
#   (CORREGIDO: marcador_pre/post con sets+juegos + gano_juego + info_break)
# ================================================================

import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Fix import cuando ejecutas desde /scripts ---
# Permite: python pipeline_golpes.py desde scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.saque_utils import (
    inferir_parejas,
    extraer_sacador,
    inferir_ganador_punto,
    etiquetar_puntos_saque,
    resumen_estadisticas_saque
)

# ================================
# CONFIG
# ================================
ANCHO_PISTA = 100
LARGO_PISTA = 200

COL_JUGADOR = "jugador"
COL_GOLPE = "golpe_q"
COL_CATEG = "categoria_punto"

# columnas alias (soluciona typos tipo “gople”)
COLUMN_ALIASES = {
    "inicio_x": ["inicio_golpe:_x", "inicio_gople:_x", "inicio_golpe_x"],
    "inicio_y": ["inicio_golpe:_y", "inicio_gople:_y", "inicio_golpe_y"],
    "fin_x":    ["fin_golpe:_x", "fin_gople:_x", "fin_golpe_x"],
    "fin_y":    ["fin_golpe:_y", "fin_gople:_y", "fin_golpe_y"],
}

COLORES_EVENTO = {
    "winner": "#00BFFF",
    "error no forzado": "#FF9800",
    "fuerza_error": "#FF3B3B",
    "bola dentro": "#4CAF50"
}

# ==========================================================
# HELPERS
# ==========================================================
def _normalize_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace(" ", "_")
    c = c.replace(":", "")
    c = c.replace("-", "_")
    c = re.sub(r"__+", "_", c)
    return c


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_colname(c) for c in df.columns]
    return df


def resolve_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Crea (si existen) las columnas canónicas inicio_x/inicio_y/fin_x/fin_y desde alias."""
    df = df.copy()
    for key, candidates in COLUMN_ALIASES.items():
        if key in df.columns:
            continue
        found = None
        for c in candidates:
            c_norm = _normalize_colname(c)
            if c_norm in df.columns:
                found = c_norm
                break
        if found is not None:
            df[key] = df[found]
    return df


def norm_name(x: str) -> str:
    """Normaliza para comparar nombres robustamente (case/espacios/acentos simples)."""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = (s.replace("á", "a").replace("é", "e").replace("í", "i")
           .replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("ñ", "n"))
    return s


def parse_score(x):
    """Parsea 'a-b' o 'a:b' y devuelve (a,b) ints. Si falla, (None,None)."""
    if pd.isna(x):
        return (None, None)
    s = str(x).strip()
    if not s:
        return (None, None)
    s = s.replace(":", "-").replace("–", "-").replace("—", "-").replace("_", "-")
    parts = s.split("-")
    if len(parts) != 2:
        return (None, None)
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except:
        return (None, None)


# ==========================================================
# 1. CARGA
# ==========================================================
def cargar_golpes(path="data/processed/golpes.parquet"):
    print(f"📂 Cargando golpes: {os.path.abspath(path)}")
    df = pd.read_parquet(path)
    df = normalizar_columnas(df)
    df = resolve_coordinate_columns(df)
    print(f"✅ {len(df)} golpes cargados.\n")
    return df


# ==========================================================
# 2. CATEGORÍA EVENTO
# ==========================================================
def clasificar_eventos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if COL_CATEG not in df.columns:
        print("⚠ Creando categoria_punto automáticamente...")

        def obtener(row):
            for col in ["error", "winner", "fuerza_error"]:
                if col in row.index:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()
            return "bola dentro"

        df[COL_CATEG] = df.apply(obtener, axis=1)

    df[COL_CATEG] = df[COL_CATEG].astype(str).str.lower().str.strip()

    df[COL_CATEG] = df[COL_CATEG].replace({
        "fuerza error": "fuerza_error",
        "fuerza_error": "fuerza_error",
    })

    return df


# ==========================================================
# 3. RECONSTRUIR SET / JUEGO / PUNTO
# ==========================================================
def reconstruir_marcadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "clip_start" in df.columns:
        df = df.sort_values("clip_start").reset_index(drop=True)

    print("🔄 Reconstruyendo marcador...")

    required = ["marcador_sets", "marcador_juegos", "marcador_puntos"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas {missing}. Necesitas marcador_sets/marcador_juegos/marcador_puntos en el parquet."
        )

    df["set_real"] = (df["marcador_sets"].shift() != df["marcador_sets"]).cumsum()
    df["juego_real"] = (
        (df["marcador_juegos"].shift() != df["marcador_juegos"]) |
        (df["set_real"].shift() != df["set_real"])
    ).cumsum()
    df["punto_real"] = (df["marcador_puntos"].shift() != df["marcador_puntos"]).cumsum()

    print("✔ Marcador reconstruido.\n")
    return df


# ==========================================================
# 4. MÉTRICAS + SUMA + MEDIA + STD
# ==========================================================
def resumen_metricas_por_jugador(df: pd.DataFrame) -> pd.DataFrame:
    tabla = df.groupby([COL_JUGADOR, COL_CATEG]).size().unstack(fill_value=0)
    tabla["total"] = tabla.sum(axis=1)

    porcentajes = tabla.div(tabla["total"], axis=0).mul(100).round(2)
    full = pd.concat([tabla, porcentajes.add_suffix("_%")], axis=1)

    full.loc["SUMA"] = full.sum(numeric_only=True)

    jugadores = full.drop(index="SUMA")
    full.loc["MEDIA"] = jugadores.mean(numeric_only=True).round(2)
    full.loc["STD"] = jugadores.std(numeric_only=True).round(2)

    return full.reset_index()


# ==========================================================
# 5. TOP GOLPES POR SET
# ==========================================================
def top_golpes_por_set(df: pd.DataFrame, output_dir: str):
    print("🎯 TOP golpes por set")
    os.makedirs(output_dir, exist_ok=True)

    if COL_GOLPE not in df.columns:
        print(f"⚠ Falta {COL_GOLPE} → se omite top golpes por set.")
        return

    for s in sorted(df["set_real"].dropna().unique()):
        df_s = df[df["set_real"] == s]
        print(f"  ➤ Set {s}: {len(df_s)} golpes")

        df_valid = df_s[~df_s[COL_GOLPE].astype(str).str.lower().isin(["saque", "servicio", "service"])]

        conteo = df_valid.groupby([COL_JUGADOR, COL_GOLPE]).size().reset_index(name="conteo")

        for jugador, dfj in conteo.groupby(COL_JUGADOR):
            df_top = dfj.sort_values("conteo", ascending=False).head(5)

            plt.figure(figsize=(6, 4))
            plt.barh(df_top[COL_GOLPE], df_top["conteo"])
            plt.title(f"Top golpes — {jugador} — Set {s}")
            plt.tight_layout()

            fname = f"top_golpes_{jugador}_set_{s}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()
            print(f"      📁 {fname}")


# ==========================================================
# 6. PISTA POR SET
# ==========================================================
def pintar_pista_por_set(df: pd.DataFrame, output_dir: str):
    print("\n🎾 Pintando pista por set...")
    os.makedirs(output_dir, exist_ok=True)

    REQUIRED = ["inicio_x", "inicio_y", "fin_x", "fin_y"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"⚠ FALTAN columnas de coordenadas: {missing} → se omite pista por set.\n")
        return

    for s in sorted(df["set_real"].dropna().unique()):
        print(f"  ➤ Set {s}")
        df_s = df[df["set_real"] == s]

        for jugador in df_s[COL_JUGADOR].dropna().unique():
            sub = df_s[df_s[COL_JUGADOR] == jugador].copy().dropna(subset=REQUIRED)
            if sub.empty:
                print(f"     ⚠ Jugador {jugador} sin coordenadas → se omite.")
                continue

            fig = go.Figure()

            SERVICIO_Y_OFFSET = 60
            CENTRO_X = ANCHO_PISTA / 2

            fig.add_shape(type="rect", x0=0, y0=0, x1=ANCHO_PISTA, y1=LARGO_PISTA,
                          line=dict(color="white", width=3))
            fig.add_shape(type="line", x0=0, y0=LARGO_PISTA / 2, x1=ANCHO_PISTA, y1=LARGO_PISTA / 2,
                          line=dict(color="white", width=4))
            fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                          x1=ANCHO_PISTA, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))
            fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                          x1=ANCHO_PISTA, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))
            fig.add_shape(type="line", x0=CENTRO_X, y0=LARGO_PISTA/2,
                          x1=CENTRO_X, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))
            fig.add_shape(type="line", x0=CENTRO_X, y0=LARGO_PISTA/2,
                          x1=CENTRO_X, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))

            traces = []
            for cat, color in COLORES_EVENTO.items():
                df_cat = sub[sub[COL_CATEG] == cat]
                if df_cat.empty:
                    continue

                first_legend = True
                for _, r in df_cat.iterrows():
                    traces.append(go.Scatter(
                        x=[r["inicio_x"], r["fin_x"]],
                        y=[r["inicio_y"], r["fin_y"]],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=6, symbol="circle"),
                        name=cat,
                        legendgroup=cat,
                        showlegend=first_legend,
                    ))
                    traces.append(go.Scatter(
                        x=[r["fin_x"]],
                        y=[r["fin_y"]],
                        mode="markers",
                        marker=dict(size=8, color=color, symbol="triangle-up"),
                        name=f"{cat}_fin",
                        showlegend=False,
                        legendgroup=cat,
                    ))
                    first_legend = False

            fig.add_traces(traces)

            fig.update_layout(
                title=f"Pista — {jugador} — Set {s}",
                height=1000,
                width=800,
                plot_bgcolor="#003C77",
                paper_bgcolor="#00244D",
                xaxis=dict(visible=False, range=[0, ANCHO_PISTA]),
                yaxis=dict(visible=False, range=[0, LARGO_PISTA], scaleanchor="x", scaleratio=1),
                font=dict(color="white"),
                legend=dict(groupclick="togglegroup", itemclick="toggle")
            )

            filename = f"pista_jugador_{jugador}_set_{s}.html"
            fig.write_html(os.path.join(output_dir, filename))
            print(f"     📁 Guardado: {filename}")


# ==========================================================
# 7. MÉTRICAS — PARTIDO / SET / JUEGO (+ info_break)
# ==========================================================
def exportar_metricas(df: pd.DataFrame, out: str, pareja1, pareja2):
    print("\n📊 Exportando métricas...")
    os.makedirs(out, exist_ok=True)

    max_set = int(df["set_real"].max())
    max_juego = int(df["juego_real"].max())
    print(f"   Sets: {max_set} | Juegos: {max_juego}")

    # RESUMEN PARTIDO
    resumen = resumen_metricas_por_jugador(df)
    resumen = resumen[~resumen[COL_JUGADOR].isin(["SUMA", "MEDIA", "STD"])]
    resumen.to_excel(os.path.join(out, "resumen_partido.xlsx"), index=False)

    # RESUMEN POR SET
    for s in range(1, max_set + 1):
        res_s = resumen_metricas_por_jugador(df[df["set_real"] == s])
        res_s = res_s[~res_s[COL_JUGADOR].isin(["SUMA", "MEDIA", "STD"])]
        res_s.to_excel(os.path.join(out, f"resumen_set_{s}.xlsx"), index=False)

    # RESUMEN POR JUEGO
    rows = []

    for j in range(1, max_juego + 1):
        df_j = df[df["juego_real"] == j].copy()
        if df_j.empty:
            continue

        set_j = df_j["set_real"].iloc[0]

        resumen_j = resumen_metricas_por_jugador(df_j)
        resumen_j = resumen_j[~resumen_j[COL_JUGADOR].isin(["SUMA", "MEDIA", "STD"])]

        # marcador_pre (sets+juegos al inicio del juego)
        ms_pre_series = df_j["marcador_sets"].dropna()
        mj_pre_series = df_j["marcador_juegos"].dropna()
        sets_pre = ms_pre_series.iloc[0] if len(ms_pre_series) else np.nan
        juegos_pre = mj_pre_series.iloc[0] if len(mj_pre_series) else np.nan

        # marcador_post (sets+juegos al final del juego) => inicio del siguiente juego si existe
        df_next = df[df["juego_real"] == (j + 1)]
        ms_next = df_next["marcador_sets"].dropna() if not df_next.empty else pd.Series([], dtype=object)
        mj_next = df_next["marcador_juegos"].dropna() if not df_next.empty else pd.Series([], dtype=object)

        if len(ms_next) and len(mj_next):
            sets_post = ms_next.iloc[0]
            juegos_post = mj_next.iloc[0]
        else:
            ms_last = df_j["marcador_sets"].dropna()
            mj_last = df_j["marcador_juegos"].dropna()
            sets_post = ms_last.iloc[-1] if len(ms_last) else np.nan
            juegos_post = mj_last.iloc[-1] if len(mj_last) else np.nan

        resumen_j["marcador_pre"] = f"sets: {sets_pre} | juegos: {juegos_pre}"
        resumen_j["marcador_post"] = f"sets: {sets_post} | juegos: {juegos_post}"
        resumen_j["set"] = set_j
        resumen_j["juego"] = j

        # info_saque
        saca_jugador = df_j.get("saca_jugador", pd.Series(dtype=object)).dropna().unique()
        saca_jugador = saca_jugador[0] if len(saca_jugador) else None

        saca_pareja = df_j.get("saca_pareja", pd.Series(dtype=object)).dropna().unique()
        saca_pareja = saca_pareja[0] if len(saca_pareja) else None

        def info_saque(jugador):
            if "pareja_id" not in df.columns:
                return np.nan
            pareja_j = df[df[COL_JUGADOR] == jugador]["pareja_id"].iloc[0]
            if saca_jugador is None:
                return np.nan
            if jugador == saca_jugador:
                return 1
            if saca_pareja is not None and pareja_j == saca_pareja:
                return 2
            return 3

        resumen_j["info_saque"] = resumen_j[COL_JUGADOR].apply(info_saque)

        # # gano_juego (pareja1-pareja2)
        # a_pre, b_pre = parse_score(juegos_pre)
        # a_post, b_post = parse_score(juegos_post)

        # ganador = None  # 1 => pareja1, 2 => pareja2
        # if None not in (a_pre, b_pre, a_post, b_post):
        #     da, db = (a_post - a_pre), (b_post - b_pre)
        #     if (da, db) == (1, 0):
        #         ganador = 1
        #     elif (da, db) == (0, 1):
        #         ganador = 2

        # ---------- determinar ganador del juego ----------
        a_pre, b_pre = parse_score(juegos_pre)
        a_post, b_post = parse_score(juegos_post)

        s1_pre, s2_pre = parse_score(sets_pre)
        s1_post, s2_post = parse_score(sets_post)

        ganador = None  # 1 => pareja1, 2 => pareja2

        # Caso 1: juego normal (no cambia el set)
        if None not in (a_pre, b_pre, a_post, b_post) and (s1_pre == s1_post and s2_pre == s2_post):
            da, db = (a_post - a_pre), (b_post - b_pre)
            if (da, db) == (1, 0):
                ganador = 1
            elif (da, db) == (0, 1):
                ganador = 2

        # Caso 2: cambio de set → el ganador del set gana el último juego
        elif None not in (s1_pre, s2_pre, s1_post, s2_post):
            ds1, ds2 = (s1_post - s1_pre), (s2_post - s2_pre)
            if (ds1, ds2) == (1, 0):
                ganador = 1
            elif (ds1, ds2) == (0, 1):
                ganador = 2


        P1_KEYS = ("chingotto", "galan")
        P2_KEYS = ("coello", "tapia")

        def pertenece_p1(nombre_norm: str) -> bool:
            return any(k in nombre_norm for k in P1_KEYS)

        def pertenece_p2(nombre_norm: str) -> bool:
            return any(k in nombre_norm for k in P2_KEYS)

        def gano_juego_por_jugador(jugador):
            jn = norm_name(jugador)
            if ganador == 1:
                return pertenece_p1(jn)
            if ganador == 2:
                return pertenece_p2(jn)
            return False

        resumen_j["gano_juego"] = resumen_j[COL_JUGADOR].apply(gano_juego_por_jugador)

        # ==========================
        # NUEVO: info_break
        # ==========================
        # Determinar pareja que saca (1 o 2)
        sacador_team = None  # 1 => pareja1, 2 => pareja2

        if saca_jugador is not None:
            sj = norm_name(saca_jugador)
            if pertenece_p1(sj):
                sacador_team = 1
            elif pertenece_p2(sj):
                sacador_team = 2
        elif saca_pareja is not None and "pareja_id" in df.columns:
            # fallback por pareja_id si tu saque_utils lo rellena así
            try:
                # pareja1/2 se asumen como ids 1/2 (si es tu caso); si no, no lo forzamos
                if int(saca_pareja) in (1, 2):
                    sacador_team = int(saca_pareja)
            except:
                sacador_team = None

        def info_break_por_jugador(jugador):
            # Si no sabemos sacador o ganador, no inventamos
            if sacador_team is None or ganador is None:
                return "desconocido"

            # Si gana el que saca => no break para todos
            if sacador_team == ganador:
                return "no_break"

            # Si pierde el que saca => hay break:
            # - Sacador: recibe_break
            # - Restador: hace_break
            jn = norm_name(jugador)
            in_sacador = pertenece_p1(jn) if sacador_team == 1 else pertenece_p2(jn)
            return "recibe_break" if in_sacador else "hace_break"

        resumen_j["info_break"] = resumen_j[COL_JUGADOR].apply(info_break_por_jugador)

        rows.append(resumen_j)
        rows.append(pd.DataFrame([[""] * len(resumen_j.columns)], columns=resumen_j.columns))

    df_juegos = pd.concat(rows, ignore_index=True)
    df_juegos.to_excel(os.path.join(out, "resumen_juegos.xlsx"), index=False)

    print("   ✔ resumen_juegos.xlsx generado con marcador_pre/post, gano_juego e info_break.\n")


# ==========================================================
# TOP GOLPES PARTIDO
# ==========================================================
def top_golpes_partido(df: pd.DataFrame, output_dir: str):
    print("🎯 TOP golpes del partido entero")
    os.makedirs(output_dir, exist_ok=True)

    if COL_GOLPE not in df.columns:
        print(f"⚠ Falta {COL_GOLPE} → se omite top golpes partido.")
        return

    df_valid = df[~df[COL_GOLPE].astype(str).str.lower().isin(["saque", "servicio", "service"])]
    conteo = df_valid.groupby([COL_JUGADOR, COL_GOLPE]).size().reset_index(name="conteo")

    for jugador, dfj in conteo.groupby(COL_JUGADOR):
        df_top = dfj.sort_values("conteo", ascending=False).head(10)

        plt.figure(figsize=(7, 5))
        plt.barh(df_top[COL_GOLPE], df_top["conteo"])
        plt.title(f"Top golpes — Partido Completo — {jugador}")
        plt.tight_layout()

        fname = f"top_golpes_partido_{jugador}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"   📁 {fname}")


# ==========================================================
# PISTA PARTIDO
# ==========================================================
def pintar_pista_partido(df: pd.DataFrame, output_dir: str):
    print("🎾 Pintando pista del PARTIDO ENTERO...")
    os.makedirs(output_dir, exist_ok=True)

    REQUIRED = ["inicio_x", "inicio_y", "fin_x", "fin_y"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"⚠ FALTAN columnas: {missing} → NO SE PUEDE PINTAR PISTA COMPLETA\n")
        return

    for jugador in df[COL_JUGADOR].dropna().unique():
        sub = df[df[COL_JUGADOR] == jugador].dropna(subset=REQUIRED).copy()
        if sub.empty:
            print(f"  ⚠ Jugador {jugador} sin golpes → omitiendo.")
            continue

        fig = go.Figure()

        SERVICIO_Y_OFFSET = 60
        CENTRO_X = ANCHO_PISTA / 2

        fig.add_shape(type="rect", x0=0, y0=0, x1=ANCHO_PISTA, y1=LARGO_PISTA,
                      line=dict(color="white", width=3))
        fig.add_shape(type="line", x0=0, y0=LARGO_PISTA / 2, x1=ANCHO_PISTA, y1=LARGO_PISTA / 2,
                      line=dict(color="white", width=4))
        fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                      x1=ANCHO_PISTA, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                      x1=ANCHO_PISTA, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=CENTRO_X, y0=LARGO_PISTA/2,
                      x1=CENTRO_X, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=CENTRO_X, y0=LARGO_PISTA/2,
                      x1=CENTRO_X, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))

        traces = []
        for cat, color in COLORES_EVENTO.items():
            df_cat = sub[sub[COL_CATEG] == cat]
            if df_cat.empty:
                continue

            first_legend = True
            for _, r in df_cat.iterrows():
                traces.append(go.Scatter(
                    x=[r["inicio_x"], r["fin_x"]],
                    y=[r["inicio_y"], r["fin_y"]],
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=6, symbol="circle"),
                    name=cat,
                    legendgroup=cat,
                    showlegend=first_legend,
                ))
                traces.append(go.Scatter(
                    x=[r["fin_x"]],
                    y=[r["fin_y"]],
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="triangle-up"),
                    name=f"{cat}_fin",
                    hoverinfo="none",
                    showlegend=False,
                    legendgroup=cat,
                ))
                first_legend = False

        fig.add_traces(traces)

        fig.update_layout(
            title=f"Pista — Partido Completo — {jugador}",
            height=900,
            width=700,
            plot_bgcolor="#003C77",
            paper_bgcolor="#00244D",
            xaxis=dict(visible=False, range=[0, ANCHO_PISTA]),
            yaxis=dict(visible=False, range=[0, LARGO_PISTA], scaleanchor="x", scaleratio=1),
            font=dict(color="white"),
            legend=dict(groupclick="togglegroup", itemclick="toggle")
        )

        filename = f"pista_partido_completo_{jugador}.html"
        fig.write_html(os.path.join(output_dir, filename))
        print(f"   📁 guardado: {filename}")


# ==========================================================
# PIPELINE COMPLETO
# ==========================================================
def analizar_partido_completo_trazado(
    ruta_golpes="data/processed/golpes.parquet",
    out_dir="outputs/analisis"
):
    print("\n========================================")
    print("🔎 INICIANDO ANÁLISIS COMPLETO")
    print("========================================\n")

    df = cargar_golpes(ruta_golpes)
    df = clasificar_eventos(df)
    df = reconstruir_marcadores(df)

    # Procesamiento de saque
    df, pareja1, pareja2 = inferir_parejas(df)  # pareja1=Chingotto-Galán, pareja2=Coello-Tapia
    df = extraer_sacador(df)
    df = inferir_ganador_punto(df)
    df = etiquetar_puntos_saque(df)

    # guardar estadísticas de saque
    os.makedirs(out_dir, exist_ok=True)
    stats_saque = resumen_estadisticas_saque(df, pareja1, pareja2)
    stats_saque.to_excel(os.path.join(out_dir, "estadisticas_saque.xlsx"), index=False)
    print("📄 Estadísticas de saque guardadas en estadisticas_saque.xlsx\n")

    # métricas + gráficos
    exportar_metricas(df, out_dir, pareja1, pareja2)
    top_golpes_por_set(df, out_dir)
    pintar_pista_por_set(df, out_dir)
    top_golpes_partido(df, out_dir)
    pintar_pista_partido(df, out_dir)

    print("\n========================================")
    print("✅ ANÁLISIS COMPLETO GENERADO EN:")
    print(os.path.abspath(out_dir))
    print("========================================\n")

    return df


if __name__ == "__main__":
    analizar_partido_completo_trazado()
