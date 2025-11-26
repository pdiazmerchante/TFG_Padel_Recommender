# ================================================================
# PIPELINE COMPLETO — reconstrucción marcador + análisis + gráficos
# ================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
def resolve_coordinate_columns(df):
    """Devuelve dict con las columnas reales existentes para coordenadas."""
    resolved = {}

    for key, candidates in COLUMN_ALIASES.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        resolved[key] = found

    return resolved


# ==========================================================
# 1. CARGA
# ==========================================================
def cargar_golpes(path="data/processed/golpes.parquet"):
    print(f"📂 Cargando golpes: {os.path.abspath(path)}")
    df = pd.read_parquet(path)
    print(f"✅ {len(df)} golpes cargados.\n")
    return df


# ==========================================================
# 2. CATEGORÍA EVENTO
# ==========================================================
def clasificar_eventos(df):
    df = df.copy()

    if COL_CATEG not in df.columns:
        print("⚠ Creando categoria_punto automáticamente...")
        def obtener(row):
            for col in ["error", "winner", "fuerza_error"]:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    return val
            return "bola dentro"

        df[COL_CATEG] = df.apply(obtener, axis=1)

    df[COL_CATEG] = df[COL_CATEG].astype(str).str.lower().str.strip()
    return df


# ==========================================================
# 3. RECONSTRUIR SET / JUEGO / PUNTO
# ==========================================================
def reconstruir_marcadores(df):
    df = df.copy()

    if "clip_start" in df.columns:
        df = df.sort_values("clip_start").reset_index(drop=True)

    print("🔄 Reconstruyendo marcador...")

    df["set_real"] = (df["marcador_sets"].shift() != df["marcador_sets"]).cumsum()
    df["juego_real"] = ((df["marcador_juegos"].shift() != df["marcador_juegos"]) |
                        (df["set_real"].shift() != df["set_real"])).cumsum()
    df["punto_real"] = (df["marcador_puntos"].shift() != df["marcador_puntos"]).cumsum()

    print("✔ Marcador reconstruido.\n")
    return df


# ==========================================================
# 4. MÉTRICAS + SUMA + MEDIA + STD
# ==========================================================
def resumen_metricas_por_jugador(df):
    tabla = df.groupby([COL_JUGADOR, COL_CATEG]).size().unstack(fill_value=0)
    tabla["total"] = tabla.sum(axis=1)

    porcentajes = tabla.div(tabla["total"], axis=0).mul(100).round(2)

    full = pd.concat([tabla, porcentajes.add_suffix("_%")], axis=1)

    # === SUMA / MEDIA / STD ===
    full.loc["SUMA"] = full.sum(numeric_only=True)
    full.loc["MEDIA"] = full.mean(numeric_only=True).round(2)
    full.loc["STD"] = full.std(numeric_only=True).round(2)

    return full.reset_index()


# ==========================================================
# 5. TOP GOLPES POR SET
# ==========================================================
def top_golpes_por_set(df, output_dir):
    print("🎯 TOP golpes por set")
    os.makedirs(output_dir, exist_ok=True)

    for s in sorted(df["set_real"].unique()):
        df_s = df[df["set_real"] == s]
        print(f"  ➤ Set {s}: {len(df_s)} golpes")

        df_valid = df_s[~df_s[COL_GOLPE].str.lower().isin(["saque", "servicio", "service"])]

        conteo = df_valid.groupby([COL_JUGADOR, COL_GOLPE]).size().reset_index(name="conteo")

        for jugador, dfj in conteo.groupby(COL_JUGADOR):
            df_top = dfj.sort_values("conteo", ascending=False).head(5)

            plt.figure(figsize=(6,4))
            plt.barh(df_top[COL_GOLPE], df_top["conteo"])
            plt.title(f"Top golpes — {jugador} — Set {s}")
            plt.tight_layout()

            fname = f"top_golpes_{jugador}_set_{s}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()
            print(f"      📁 {fname}")


# ==========================================================
# 6. PISTA POR SET (robusta con alias)
# ==========================================================
def pintar_pista_por_set(df, output_dir):

    import os
    import plotly.graph_objects as go

    print("\n🎾 Pintando pista por set...")
    os.makedirs(output_dir, exist_ok=True)

    REQUIRED = ["inicio_x", "inicio_y", "fin_x", "fin_y"]

    # Comprobación columnas
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"⚠ FALTAN columnas de coordenadas: {missing}")
        print("   → No se pintarán golpes sin coordenadas válidas.\n")

    for s in sorted(df["set_real"].dropna().unique()):
        print(f"  ➤ Set {s}")

        df_s = df[df["set_real"] == s]

        for jugador in df_s["jugador"].dropna().unique():

            sub = df_s[df_s["jugador"] == jugador].copy()

            # Filtrar solo golpes con las 4 coordenadas
            sub = sub.dropna(subset=REQUIRED)

            if sub.empty:
                print(f"     ⚠ Jugador {jugador} sin golpes con coordenadas → se omite.")
                continue

            fig = go.Figure()

            # ============================
            # 🎾 DIBUJO DE LA PISTA
            # ============================
            SERVICIO_Y_OFFSET = 60
            CENTRO_X = ANCHO_PISTA / 2

            fig.add_shape(type="rect", x0=0, y0=0,
                          x1=ANCHO_PISTA, y1=LARGO_PISTA,
                          line=dict(color="white", width=3))

            fig.add_shape(type="line",
                          x0=0, y0=LARGO_PISTA / 2,
                          x1=ANCHO_PISTA, y1=LARGO_PISTA / 2,
                          line=dict(color="white", width=4))

            fig.add_shape(type="line",
                          x0=0, y0=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                          x1=ANCHO_PISTA, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))

            fig.add_shape(type="line",
                          x0=0, y0=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                          x1=ANCHO_PISTA, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))

            fig.add_shape(type="line",
                          x0=CENTRO_X, y0=LARGO_PISTA/2,
                          x1=CENTRO_X, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))

            fig.add_shape(type="line",
                          x0=CENTRO_X, y0=LARGO_PISTA/2,
                          x1=CENTRO_X, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                          line=dict(color="white", width=2))

            # ==================================================
            # 🔥 TRAZAS (Línea + inicio círculo + fin triángulo)
            # ==================================================
            traces = []

            for cat, color in COLORES_EVENTO.items():

                df_cat = sub[sub["categoria_punto"] == cat]

                if df_cat.empty:
                    continue

                first_legend = True

                for _, r in df_cat.iterrows():

                    # ---- LÍNEA + INICIO (círculo) ----
                    traces.append(go.Scatter(
                        x=[r["inicio_x"], r["fin_x"]],
                        y=[r["inicio_y"], r["fin_y"]],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=6, symbol="circle"),
                        name=cat,
                        legendgroup=cat,
                        showlegend=first_legend,
                        hovertext=f"{cat} (trayectoria)",
                        hoverinfo="text",
                    ))

                    # ---- SOLO FIN (triángulo) ----
                    traces.append(go.Scatter(
                        x=[r["fin_x"]],
                        y=[r["fin_y"]],
                        mode="markers",
                        marker=dict(size=8, color=color, symbol="triangle-up"),
                        name=f"{cat} fin",
                        showlegend=False,
                        legendgroup=cat,
                        hovertext=f"{cat} (fin)",
                        hoverinfo="text",
                    ))

                    first_legend = False

            fig.add_traces(traces)

            # ============================
            # 🎨 LAYOUT
            # ============================
            fig.update_layout(
                title=f"Pista — {jugador} — Set {s}",
                height=1000,
                width=800,
                plot_bgcolor="#003C77",
                paper_bgcolor="#00244D",
                xaxis=dict(visible=False, range=[0, ANCHO_PISTA]),
                yaxis=dict(visible=False, range=[0, LARGO_PISTA],
                           scaleanchor="x",
                           scaleratio=1),
                font=dict(color="white"),
                legend=dict(
                    groupclick="togglegroup",
                    itemclick="toggle"
                )
            )

            filename = f"pista_jugador_{jugador}_set_{s}.html"
            fig.write_html(os.path.join(output_dir, filename))

            print(f"     📁 Guardado: {filename}")

# ==========================================================
# 7. MÉTRICAS — PARTIDO / SET / JUEGO (1 archivo juegos)
# ==========================================================
def exportar_metricas(df, out):
    print("\n📊 Exportando métricas...")

    os.makedirs(out, exist_ok=True)

    max_set = df["set_real"].max()
    max_juego = df["juego_real"].max()

    print(f"   Sets: {max_set} | Juegos: {max_juego}")

    # --- Partido ---
    resumen = resumen_metricas_por_jugador(df)
    resumen.to_excel(os.path.join(out, "resumen_partido.xlsx"), index=False)

    # --- Sets ---
    for s in range(1, max_set+1):
        res_s = resumen_metricas_por_jugador(df[df["set_real"] == s])
        res_s.to_excel(os.path.join(out, f"resumen_set_{s}.xlsx"), index=False)

    # --- Juegos: UN SOLO ARCHIVO ---
    writer = pd.ExcelWriter(os.path.join(out, "resumen_juegos.xlsx"), engine="xlsxwriter")

    for j in range(1, max_juego+1):
        res_j = resumen_metricas_por_jugador(df[df["juego_real"] == j])
        res_j.to_excel(writer, sheet_name=f"Juego_{j}", index=False)

    writer.close()
    print("   ✔ resumen_juegos.xlsx generado\n")

def top_golpes_partido(df, output_dir):
    print("🎯 TOP golpes del partido entero")

    os.makedirs(output_dir, exist_ok=True)

    # Filtrar golpes válidos (no saque)
    df_valid = df[~df[COL_GOLPE].str.lower().isin(["saque", "servicio", "service"])]

    # Conteo por jugador y tipo de golpe
    conteo = df_valid.groupby([COL_JUGADOR, COL_GOLPE]).size().reset_index(name="conteo")

    for jugador, dfj in conteo.groupby(COL_JUGADOR):
        df_top = dfj.sort_values("conteo", ascending=False).head(10)

        plt.figure(figsize=(7,5))
        plt.barh(df_top[COL_GOLPE], df_top["conteo"])
        plt.title(f"Top golpes — Partido Completo — {jugador}")
        plt.tight_layout()

        fname = f"top_golpes_partido_{jugador}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"   📁 {fname}")

def pintar_pista_partido(df, output_dir):

    print("🎾 Pintando pista del PARTIDO ENTERO...")
    os.makedirs(output_dir, exist_ok=True)

    REQUIRED = ["inicio_x", "inicio_y", "fin_x", "fin_y"]

    # Verificar columnas
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

        # 🎾 DIBUJO DE LA PISTA (igual que en tu función anterior)
        SERVICIO_Y_OFFSET = 60
        CENTRO_X = ANCHO_PISTA / 2

        fig.add_shape(type="rect", x0=0, y0=0,
                      x1=ANCHO_PISTA, y1=LARGO_PISTA,
                      line=dict(color="white", width=3))

        fig.add_shape(type="line",
                      x0=0, y0=LARGO_PISTA / 2,
                      x1=ANCHO_PISTA, y1=LARGO_PISTA / 2,
                      line=dict(color="white", width=4))

        fig.add_shape(type="line",
                      x0=0, y0=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                      x1=ANCHO_PISTA, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))

        fig.add_shape(type="line",
                      x0=0, y0=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                      x1=ANCHO_PISTA, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))

        fig.add_shape(type="line",
                      x0=CENTRO_X, y0=LARGO_PISTA/2,
                      x1=CENTRO_X, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))

        fig.add_shape(type="line",
                      x0=CENTRO_X, y0=LARGO_PISTA/2,
                      x1=CENTRO_X, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET,
                      line=dict(color="white", width=2))

        # 🔥 Trayectorias para TODO el partido
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
            height=900,          # 👈 MÁS ALTO
            width=700,           # 👈 MÁS ANCHO
            plot_bgcolor="#003C77",
            paper_bgcolor="#00244D",
            xaxis=dict(visible=False, range=[0, ANCHO_PISTA]),
            yaxis=dict(visible=False, range=[0, LARGO_PISTA],
                       scaleanchor="x", scaleratio=1),
            font=dict(color="white"),
            legend=dict(groupclick="togglegroup", itemclick="toggle")
        )

        filename = f"pista_partido_completo_{jugador}.html"
        fig.write_html(os.path.join(output_dir, filename))

        print(f"   📁 guardado: {filename}")



# ==========================================================
# 8. PIPELINE COMPLETO
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

    exportar_metricas(df, out_dir)
    top_golpes_por_set(df, out_dir)
    pintar_pista_por_set(df, out_dir)
    top_golpes_partido(df, out_dir)
    pintar_pista_partido(df, out_dir)

    # =============================
    # 💥 NUEVO: procesamiento saque
    # =============================
    df, pareja1, pareja2 = inferir_parejas(df)
    df = extraer_sacador(df)
    df = inferir_ganador_punto(df)
    df = etiquetar_puntos_saque(df)

    # Guardar CSV de estadísticas de saque
    stats_saque = resumen_estadisticas_saque(df, pareja1, pareja2)
    stats_saque.to_excel(os.path.join(out_dir, "estadisticas_saque.xlsx"), index=False)

    print("📄 Estadísticas de saque guardadas en estadisticas_saque.xlsx")



    print("\n========================================")
    print("✅ ANÁLISIS COMPLETO GENERADO EN:")
    print(os.path.abspath(out_dir))
    print("========================================\n")

    return df


if __name__ == "__main__":
    analizar_partido_completo_trazado()
