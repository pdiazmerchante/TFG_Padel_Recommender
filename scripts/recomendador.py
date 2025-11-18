# recomendador_nivel2.py

import os
import numpy as np
import pandas as pd

# Reutilizamos constantes del pipeline
from pipeline_juegos import ANCHO_PISTA, LARGO_PISTA, COL_INICIO_X, COL_FIN_X, COL_JUGADOR, COL_FIN_Y

# ======================================================
# CONFIGURACIÓN
# ======================================================

# 👇 Rellena aquí los nombres de tus jugadores (los tuyos)
NUESTROS_JUGADORES = ["Jorge Nieto","Miguel Yanguas"]  # <-- ajusta según dataset

MIN_EVENTOS_ZONA = 10   # mínimo de eventos para considerar una zona “fiable”
MIN_EVENTOS_GOLPE = 5   # mínimo de eventos para considerar un golpe “fiable”

# ======================================================
# CARGA DE DATOS
# ======================================================

def cargar_eventos(out_dir):
    """
    Lee el archivo eventos_recortados.* generado por pipeline_juegos.py
    (idealmente con TODO el partido, no solo el recorte).
    """
    parquet_path = os.path.join(out_dir, "eventos_completos.parquet")
    csv_path     = os.path.join(out_dir, "eventos_completos.csv")

    if os.path.exists(parquet_path):
        print(f"📂 Cargando eventos desde {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"📂 Cargando eventos desde {csv_path}")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("No se encontró eventos_completos.parquet ni .csv en esa carpeta.")


# ======================================================
# SEPARAR NUESTROS GOLPES Y LOS DEL RIVAL
# ======================================================

def separar_nuestros_y_rivales(df, nuestros=NUESTROS_JUGADORES):
    """
    Divide el DF en nuestros golpes y golpes de los rivales
    asumiendo que 'jugador' contiene nombres.
    """
    df_nuestros = df[df[COL_JUGADOR].isin(nuestros)].copy()
    df_rivales  = df[~df[COL_JUGADOR].isin(nuestros)].copy()
    return df_nuestros, df_rivales


# ======================================================
# ZONAS ESPACIALES
# ======================================================

def agregar_zona_rival(df):
    """
    Añade columna 'zona_rival' (Izquierda/Derecha) según coordenada X de fin de golpe.
    """
    df = df.copy()
    mitad = ANCHO_PISTA / 2
    df["zona_rival"] = np.where(df[COL_FIN_X] < mitad, "Izquierda", "Derecha")
    return df

def agregar_zona_profundidad(df):
    """
    Añade columna 'zona_profundidad' (Fondo / Media / Red) según coordenada Y de fin de golpe.
    """
    df = df.copy()
    tercio = LARGO_PISTA / 3
    y = df[COL_FIN_Y]

    condiciones = [
        y <= tercio,
        (y > tercio) & (y <= 2*tercio),
        y > 2*tercio
    ]
    zonas = ["Fondo", "Media pista", "Cerca de red"]
    df["zona_profundidad"] = np.select(condiciones, zonas, default="Desconocida")
    return df

def aplicar_zonas(df):
    """
    Aplica zonas horizontales (Izq/Der) y verticales (Fondo/Media/Red).
    """
    df_z = agregar_zona_rival(df)
    df_z = agregar_zona_profundidad(df_z)
    return df_z


# ======================================================
# STATS DEL RIVAL
# ======================================================

def stats_rival(df_rivales):
    """
    Estadísticas resumen de los rivales:
    - errores por jugador
    - winners por jugador
    - distribución por categoría
    - errores por zona_profundidad
    """
    stats = {}

    # Errores y winners por jugador
    errores = df_rivales[df_rivales["categoria"] == "error no forzado"] \
                    .groupby(COL_JUGADOR).size().sort_values(ascending=False)
    winners = df_rivales[df_rivales["categoria"] == "winner"] \
                    .groupby(COL_JUGADOR).size().sort_values(ascending=False)
    categoria_global = df_rivales["categoria"].value_counts(normalize=True) * 100

    stats["errores_por_jugador"] = errores
    stats["winners_por_jugador"] = winners
    stats["distribucion_categorias_%"] = categoria_global.round(2)

    # Errores por zona de profundidad (dónde fallan más)
    df_z = agregar_zona_profundidad(df_rivales)
    errores_zona = df_z[df_z["categoria"] == "error no forzado"] \
                        .groupby("zona_profundidad").size().sort_values(ascending=False)
    stats["errores_por_zona_profundidad"] = errores_zona

    return stats


# ======================================================
# MATCHUP POR ZONA (IZQ / DER) Y PROFUNDIDAD
# ======================================================

def matchup_por_zona_lateral(df_nuestros):
    """
    Calcula winrate por zona objetivo (Izquierda/Derecha) según 'zona_rival'.
    """
    df_z = agregar_zona_rival(df_nuestros)
    tabla = df_z.groupby(["zona_rival", "categoria"]).size().unstack(fill_value=0)
    tabla["total_eventos"] = tabla.sum(axis=1)
    tabla["winrate_aprox"] = tabla.get("winner", 0) / tabla["total_eventos"].replace(0, np.nan)
    return tabla

def matchup_por_profundidad(df_nuestros):
    """
    Calcula efectividad por zona de profundidad basándose en porcentaje de winners y ENF.
    """
    df_z = agregar_zona_profundidad(df_nuestros)
    tabla = df_z.groupby(["zona_profundidad", "categoria"]).size().unstack(fill_value=0)
    tabla["total_eventos"] = tabla.sum(axis=1)

    winners = tabla.get("winner", 0)
    enf     = tabla.get("error no forzado", 0)

    tabla["pct_winner"] = winners / tabla["total_eventos"].replace(0, np.nan)
    tabla["pct_enf"]    = enf     / tabla["total_eventos"].replace(0, np.nan)

    return tabla


# ======================================================
# GOLPES PROPIOS: MEJORES Y PEORES
# ======================================================

def top_golpes_efectivos(df_nuestros, top_n=3):
    """
    Devuelve los golpes propios que más se asocian a winners.
    Requiere columna 'golpe_q'.
    """
    if "golpe_q" not in df_nuestros.columns:
        return None
    
    winners = df_nuestros[df_nuestros["categoria"] == "winner"]
    if winners.empty:
        return None

    conteo = winners.groupby("golpe_q").size()
    conteo = conteo[conteo >= MIN_EVENTOS_GOLPE]  # filtramos golpes con pocos eventos
    if conteo.empty:
        return None

    return conteo.sort_values(ascending=False).head(top_n)

def top_golpes_problematicos(df_nuestros, top_n=3):
    """
    Golpes propios que más errores no forzados generan.
    """
    if "golpe_q" not in df_nuestros.columns:
        return None
    
    enf = df_nuestros[df_nuestros["categoria"] == "error no forzado"]
    if enf.empty:
        return None

    conteo = enf.groupby("golpe_q").size()
    conteo = conteo[conteo >= MIN_EVENTOS_GOLPE]
    if conteo.empty:
        return None

    return conteo.sort_values(ascending=False).head(top_n)


# ======================================================
# GENERADOR DE RECOMENDACIONES (TEXTO)
# ======================================================

def generar_recomendacion_texto(stats_r,
                                matchup_lateral,
                                matchup_profundidad,
                                top_golpes_eff,
                                top_golpes_bad):

    lineas = []

    # ============================================
    # 1) Rival más débil (jugador con más ENF)
    # ============================================
    errores = stats_r.get("errores_por_jugador", pd.Series(dtype=int))
    if not errores.empty:
        jugador_debil = errores.index[0]
        n_errores = int(errores.iloc[0])
        lineas.append(f"Objetivo táctico: el jugador rival que más errores no forzados comete es {jugador_debil} ({n_errores} ENF).")
        lineas.append(f"Recomendación: cargar más el juego sobre {jugador_debil}, especialmente en peloteos largos para forzar fallo.\n")

    # ============================================
    # 2) Zonas LATERALES (Izquierda / Derecha)
    # ============================================
    if matchup_lateral is not None and not matchup_lateral.empty:
        ml = matchup_lateral.copy()
        ml = ml[ml["total_eventos"] >= MIN_EVENTOS_ZONA]

        if "winrate_aprox" in ml.columns and not ml.empty:
            mejor_zona = ml["winrate_aprox"].idxmax()
            mejor_wr = ml["winrate_aprox"].max() * 100

            lineas.append(f"Zona lateral más rentable: al jugar hacia la zona {mejor_zona}, vuestro porcentaje de intercambios favorables ronda el {mejor_wr:.1f}%.")
            lineas.append(f"Recomendación: priorizar esa orientación del golpe, sobre todo para iniciar ataques.\n")

    # ============================================
    # 3) PROFUNDIDAD (Fondo / Media / Red)
    # Usamos IMPACTO NETO = pct_winner - pct_enf
    # ============================================
    if matchup_profundidad is not None and not matchup_profundidad.empty:

        mp = matchup_profundidad.copy()
        mp = mp[mp["total_eventos"] >= MIN_EVENTOS_ZONA]
        mp = mp[mp.index != "Desconocida"]

        if not mp.empty:
            mp["impacto_neto"] = mp["pct_winner"] - mp["pct_enf"]

            zona_mejor = mp["impacto_neto"].idxmax()
            impacto_mejor = mp["impacto_neto"].max() * 100

            zona_peor = mp["impacto_neto"].idxmin()
            impacto_peor = mp["impacto_neto"].min() * 100

            lineas.append(f"Zona de profundidad más productiva: {zona_mejor}, donde vuestro impacto neto en el punto es aproximadamente del {impacto_mejor:.1f}%.")
            lineas.append(f"Recomendación: buscar más golpes hacia {zona_mejor}, donde conseguís ventaja.\n")

            if zona_peor != zona_mejor and impacto_peor < 0:
                lineas.append(f"Zona más peligrosa: {zona_peor}, donde acumuláis más pérdidas de iniciativa (impacto negativo de {impacto_peor:.1f}%).")
                lineas.append(f"Recomendación: evitar arriesgar desde {zona_peor}, salvo que estéis muy cómodos.\n")

    # ============================================
    # 4) Golpes propios más fuertes (winners)
    # ============================================
    if top_golpes_eff is not None and len(top_golpes_eff) > 0:
        lineas.append("Golpes propios que más puntos directos generan:")
        for golpe, n in top_golpes_eff.items():
            lineas.append(f" - {golpe}: {n} puntos ganados directamente.")

        golpe_top = top_golpes_eff.index[0]
        lineas.append(f"Recomendación: buscar situaciones que favorezcan usar {golpe_top} cuando tenéis iniciativa.\n")

    # ============================================
    # 5) Golpes propios más problemáticos (errores)
    # ============================================
    if top_golpes_bad is not None and len(top_golpes_bad) > 0:
        lineas.append("Golpes que más errores no forzados generan:")
        for golpe, n in top_golpes_bad.items():
            lineas.append(f" - {golpe}: {n} errores no forzados.")

        golpe_malo = top_golpes_bad.index[0]
        lineas.append(f"Recomendación: reducir el riesgo o entrenar específicamente el golpe {golpe_malo} para mejorar la consistencia.\n")

    # ============================================
    # 6) Perfil del rival (clasificación automática)
    # ============================================
    dist = stats_r.get("distribucion_categorias_%", pd.Series(dtype=float))

    if not dist.empty:
        pct_enf = dist.get("error no forzado", 0)
        pct_winner = dist.get("winner", 0)
        pct_bola = dist.get("bola dentro", 0)

        if pct_enf <= 5 and pct_bola >= 70:
            estilo = "muy sólido y conservador"
            recomendacion_estilo = "variar alturas, abrir ángulos y cambiar ritmos para obligarles a jugar incómodos."
        elif pct_enf >= 12:
            estilo = "fallón"
            recomendacion_estilo = "mantener la bola en juego sin precipitarse para que acumulen errores."
        elif pct_winner >= 20 and 5 < pct_enf < 12:
            estilo = "ofensivo con riesgo"
            recomendacion_estilo = "jugar bolas bajas y profundas, además de evitar dejar pelotas altas que favorezcan su ataque."
        else:
            estilo = "equilibrado"
            recomendacion_estilo = "alternar fases de control y de agresividad según el momento del partido."

        lineas.append(f"Perfil del rival: {estilo} (bolas dentro {pct_bola:.1f}%, winners {pct_winner:.1f}%, ENF {pct_enf:.1f}%).")
        lineas.append(f"Recomendación: {recomendacion_estilo}\n")

    # ============================================
    # 7) Debilidad espacial del rival (zona donde más falla)
    # ============================================
    errores_zona = stats_r.get("errores_por_zona_profundidad", pd.Series(dtype=int))

    if not errores_zona.empty:
        errores_zona = errores_zona[errores_zona.index != "Desconocida"]
        errores_zona = errores_zona[errores_zona >= MIN_EVENTOS_ZONA]

        if not errores_zona.empty:
            zona_debil = errores_zona.idxmax()
            cant = errores_zona.max()

            lineas.append(f"Debilidad espacial del rival: fallan más desde {zona_debil} ({cant} ENF).")
            lineas.append(f"Recomendación: forzar que jueguen más desde {zona_debil} mediante bolas profundas o variaciones de altura.\n")

    # ============================================
    # Si no hay recomendaciones útiles
    # ============================================
    if not lineas:
        lineas.append("No se han podido generar recomendaciones tácticas con los datos disponibles.")

    return "\n".join(lineas)




# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def recomendar_estrategia(out_dir):
    df = cargar_eventos(out_dir)

    print("=== Jugadores detectados en el dataset ===")
    print(df[COL_JUGADOR].unique())
    print("=========================================\n")

    # Separar nuestros golpes vs rivales
    df_nuestros, df_rivales = separar_nuestros_y_rivales(df)

    if df_nuestros.empty or df_rivales.empty:
        print("⚠️ No se han podido separar correctamente nuestros jugadores y rivales. Revisa NUESTROS_JUGADORES.")
        return

    # Aplicar zonas
    df_nuestros_z = aplicar_zonas(df_nuestros)
    df_rivales_z  = aplicar_zonas(df_rivales)

    # Stats rival
    stats_r = stats_rival(df_rivales_z)

    # Matchups espaciales
    matchup_lateral      = matchup_por_zona_lateral(df_nuestros_z)
    matchup_profundidad  = matchup_por_profundidad(df_nuestros_z)

    # Golpes efectivos y problemáticos
    top_eff = top_golpes_efectivos(df_nuestros_z, top_n=3)
    top_bad = top_golpes_problematicos(df_nuestros_z, top_n=3)

    # Generar texto de recomendaciones
    texto = generar_recomendacion_texto(
        stats_r=stats_r,
        matchup_lateral=matchup_lateral,
        matchup_profundidad=matchup_profundidad,
        top_golpes_eff=top_eff,
        top_golpes_bad=top_bad
    )

    # Guardar recomendaciones
    rec_path = os.path.join(out_dir, "recomendaciones_nivel2.txt")
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"✅ Recomendaciones guardadas en: {os.path.abspath(rec_path)}")

    print("\n📋 Recomendaciones generadas:\n")
    print(texto)


if __name__ == "__main__":
    ruta_out = input("📁 Carpeta de salida del partido (out_dir de pipeline_juegos): ").strip()
    recomendar_estrategia(ruta_out)
