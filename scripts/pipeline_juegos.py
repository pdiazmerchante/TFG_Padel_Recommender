import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================

ANCHO_PISTA = 100
LARGO_PISTA = 200

COL_JUGADOR  = "jugador"
COL_WINNER   = "winner"
COL_ERROR    = "error"
COL_INICIO_X = "inicio_gople:_x"
COL_INICIO_Y = "inicio_gople:_y"
COL_FIN_X    = "fin_golpe:_x"
COL_FIN_Y    = "fin_golpe:_y"

COLORES_EVENTO = {
    "winner": "#00BFFF",           # Azul brillante
    "error no forzado": "#FF9800", # Naranja
    "missed": "#FF3B3B",           # Rojo
    "bola dentro": "#4CAF50"       # Verde
}


# ======================================================
# FUNCIONES AUXILIARES
# ======================================================

def safe_int(val):
    try:
        if pd.isna(val):
            return 0
        return int(val)
    except Exception:
        return 0


def cargar_datos(ruta=None):
    if ruta is None or ruta.strip() == "":
        ruta = os.path.join(os.path.dirname(__file__), "..", "data", "interim", "final_clean.parquet")
        ruta = os.path.abspath(ruta)
    print(f"📂 Cargando datos desde: {ruta}")
    df = pd.read_parquet(ruta)
    print(f"✅ Datos cargados: {len(df):,} filas.")
    return df


# ======================================================
# RECORTE Y MARCADOR
# ======================================================

def recortar_por_marcador(df, limite_juegos):
    df = df.sort_values("clip_start").reset_index(drop=True)
    df["juego_p1"] = pd.to_numeric(df["juego_p1"], errors="coerce").fillna(0).astype(int)
    df["juego_p2"] = pd.to_numeric(df["juego_p2"], errors="coerce").fillna(0).astype(int)

    idx_fin = None
    for i, row in df.iterrows():
        if (row["juego_p1"] + row["juego_p2"]) >= limite_juegos:
            idx_fin = i
            break
    if idx_fin is None:
        idx_fin = len(df) - 1

    return df.loc[:idx_fin].reset_index(drop=True)


def calcular_marcador_y_sets(df):
    set_p1 = set_p2 = 0
    sets_resultados = []

    for i in range(1, len(df)):
        j1, j2 = int(df.loc[i, "juego_p1"]), int(df.loc[i, "juego_p2"])
        if (j1 >= 6 or j2 >= 6) and abs(j1 - j2) >= 2:
            ganador = "p1" if j1 > j2 else "p2"
            if ganador == "p1":
                set_p1 += 1
            else:
                set_p2 += 1
            sets_resultados.append((j1, j2))

    ult = df.iloc[-1]
    marcador = {
        "set_p1": set_p1,
        "set_p2": set_p2,
        "juego_p1": safe_int(ult["juego_p1"]),
        "juego_p2": safe_int(ult["juego_p2"]),
    }
    return marcador, sets_resultados


# ======================================================
# CLASIFICAR EVENTOS
# ======================================================

def clasificar_eventos(df):
    d = df.copy()
    for c in [COL_WINNER, COL_ERROR]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].astype(str).str.lower().str.strip()

    d["categoria"] = "bola dentro"
    d.loc[d[COL_ERROR].str.contains("error no forzado", na=False), "categoria"] = "error no forzado"
    d.loc[d[COL_ERROR].str.contains("missed", na=False), "categoria"] = "missed"
    d.loc[d[COL_WINNER].str.contains("winner", na=False), "categoria"] = "winner"
    return d


# ======================================================
# MÉTRICAS
# ======================================================

def resumen_metricas_por_jugador(df):
    conteos = df.groupby(["jugador", "categoria"]).size().unstack(fill_value=0)
    conteos["total"] = conteos.sum(axis=1)
    porcentajes = conteos.div(conteos["total"], axis=0).mul(100).round(1)
    resumen = pd.concat([conteos, porcentajes.add_suffix("_%")], axis=1).reset_index()
    return resumen


# ======================================================
# TOP 5 GOLPES (GRÁFICAS ESTÁTICAS)
# ======================================================

def top_golpes_por_jugador(df, output_dir, top_n=5):
    d = df.copy()
    d = d[d["golpe_q"].notna()]
    conteo = d.groupby(["jugador", "golpe_q"]).size().reset_index(name="conteo")
    top = conteo.sort_values(["jugador", "conteo"], ascending=[True, False]).groupby("jugador").head(top_n)

    os.makedirs(output_dir, exist_ok=True)

    for jug, g in top.groupby("jugador"):
        plt.figure(figsize=(6, 4))
        plt.barh(g["golpe_q"], g["conteo"], color="#2196F3")
        plt.title(f"Top {top_n} golpes — {jug}")
        plt.xlabel("Repeticiones")
        plt.tight_layout()
        path = os.path.join(output_dir, f"top_golpes_{jug}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📊 Guardado: {path}")


# ======================================================
# DIRECTORIO DE SALIDA
# ======================================================

def build_output_dir(base_dir, n_juegos, marcador, set_results, df):
    os.makedirs(base_dir, exist_ok=True)
    dataset_name = "desconocido"
    if not df.empty:
        val = str(df.iloc[0, -1]).strip()
        if val and val.lower() not in ("nan", "none", "<na>"):
            dataset_name = os.path.splitext(val)[0]

    sets_text = ",".join([f"{a}-{b}" for a, b in set_results]) if set_results else ""
    j1 = safe_int(marcador.get("juego_p1", 0))
    j2 = safe_int(marcador.get("juego_p2", 0))
    if j1 > 0 or j2 > 0:
        sets_text = f"{sets_text},{j1}-{j2}" if sets_text else f"{j1}-{j2}"
    folder_name = sets_text if sets_text else f"sin_sets__{n_juegos}_juegos"
    path = os.path.join(base_dir, dataset_name, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


# ======================================================
# VISUALIZACIÓN INTERACTIVA PLOTLY
# ======================================================

def pintar_pista_interactiva(df, output_dir="outputs/html_pistas"):
    os.makedirs(output_dir, exist_ok=True)
    jugadores = df[COL_JUGADOR].dropna().unique()

    for jugador in jugadores:
        sub = df[df[COL_JUGADOR] == jugador].dropna(
            subset=[COL_INICIO_X, COL_INICIO_Y, COL_FIN_X, COL_FIN_Y]
        )
        if sub.empty:
            print(f"⚠️ No hay datos válidos para {jugador}.")
            continue

        fig = go.Figure()

        # --- Dibujo base de la pista ---
        fig.add_shape(type="rect", x0=0, y0=0, x1=ANCHO_PISTA, y1=LARGO_PISTA,
                      line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=0, y0=100, x1=ANCHO_PISTA, y1=100,
                      line=dict(color="white", width=3))
        fig.add_shape(type="line", x0=0, y0=65, x1=ANCHO_PISTA, y1=65,
                      line=dict(color="white", width=1))
        fig.add_shape(type="line", x0=0, y0=135, x1=ANCHO_PISTA, y1=135,
                      line=dict(color="white", width=1))
        fig.add_shape(type="line", x0=50, y0=65, x1=50, y1=135,
                      line=dict(color="white", width=1))

        # --- Dibujar trayectorias con inicio y fin ---
        for cat, color in COLORES_EVENTO.items():
            df_cat = sub[sub["categoria"] == cat]
            if df_cat.empty:
                continue

            for _, r in df_cat.iterrows():
                x0, y0, x1, y1 = r[COL_INICIO_X], r[COL_INICIO_Y], r[COL_FIN_X], r[COL_FIN_Y]
                distancia = ((x1 - x0)**2 + (y1 - y0)**2)**0.5

                # Línea del golpe
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=cat,
                    legendgroup=cat,
                    hoverinfo="text",
                    hovertext=f"{cat.title()}<br>"
                              f"Inicio: ({x0:.1f},{y0:.1f})<br>"
                              f"Fin: ({x1:.1f},{y1:.1f})<br>"
                              f"Distancia: {distancia:.1f} unidades",
                    showlegend=False
                ))

                # Punto de inicio
                fig.add_trace(go.Scatter(
                    x=[x0], y=[y0],
                    mode="markers",
                    marker=dict(size=6, color=color, symbol="circle"),
                    name=f"{cat} - inicio",
                    legendgroup=cat,
                    hovertext=f"Inicio ({x0:.1f},{y0:.1f})",
                    showlegend=False
                ))

                # Punto de fin con flecha/triángulo
                fig.add_trace(go.Scatter(
                    x=[x1], y=[y1],
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="triangle-up"),
                    name=f"{cat} - fin",
                    legendgroup=cat,
                    hovertext=f"Fin ({x1:.1f},{y1:.1f})",
                    showlegend=False
                ))

        # --- Controles interactivos ---
        botones = [dict(
            label="Todos", method="update",
            args=[{"visible": [True] * len(fig.data)},
                  {"title": f"Pista — {jugador} (Todos los golpes)"}]
        )]
        cats = list(COLORES_EVENTO.keys())
        for cat in cats:
            visibles = [(cat in d.name) for d in fig.data]
            botones.append(dict(
                label=cat.title(), method="update",
                args=[{"visible": visibles},
                      {"title": f"Pista — {jugador} ({cat.title()})"}]
            ))

        fig.update_layout(
            title=f"Pista interactiva — {jugador}",
            plot_bgcolor="#003C77",
            paper_bgcolor="#00244D",
            xaxis=dict(visible=False, range=[0, ANCHO_PISTA]),
            yaxis=dict(visible=False, range=[0, LARGO_PISTA], scaleanchor="x", scaleratio=1),
            updatemenus=[dict(buttons=botones, direction="down", showactive=True, x=1.15, xanchor="left", y=1.1)],
            font=dict(color="white")
        )

        out_path = os.path.join(output_dir, f"pista_{jugador}.html")
        fig.write_html(out_path)
        print(f"💾 Guardada versión interactiva con coordenadas: {out_path}")



# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def analizar_partido_interactivo():
    ruta = input("📂 Ruta del archivo parquet (Enter para usar la predeterminada): ").strip()
    df0 = cargar_datos(ruta)
    n_juegos = int(input("🎮 ¿Cuántos juegos quieres analizar?: ").strip())

    df_rec = recortar_por_marcador(df0, n_juegos)
    marcador, sets = calcular_marcador_y_sets(df_rec)
    print(f"🎯 Marcador parcial — Juegos: {marcador['juego_p1']}-{marcador['juego_p2']} | Sets: {marcador['set_p1']}-{marcador['set_p2']}")

    base = os.path.join("outputs", "figures")
    out_dir = build_output_dir(base, n_juegos, marcador, sets, df0)
    fig_dir = out_dir
    print(f"📁 Resultados guardados en: {os.path.abspath(out_dir)}")

    print("\n📊 Calculando métricas resumen...")
    df_rec = clasificar_eventos(df_rec)
    resumen = resumen_metricas_por_jugador(df_rec)
    resumen.to_csv(os.path.join(out_dir, "resumen_metricas.csv"), index=False)
    print(f"✅ Resumen guardado en {out_dir}")

    print("\n📈 Generando top golpes...")
    top_golpes_por_jugador(df_rec, output_dir=out_dir)

    print("\n🎾 Generando pistas interactivas...")
    pintar_pista_interactiva(df_rec, output_dir=fig_dir)

    print("\n✅ Análisis completo.")
    return df_rec, resumen, marcador, out_dir


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    analizar_partido_interactivo()
