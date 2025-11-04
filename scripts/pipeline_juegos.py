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
COL_INICIO_X = "inicio_golpe:_x"  # corregido
COL_INICIO_Y = "inicio_golpe:_y"  # corregido
COL_FIN_X    = "fin_golpe:_x"
COL_FIN_Y    = "fin_golpe:_y"

COLORES_EVENTO = {
    "winner": "#00BFFF",
    "error no forzado": "#FF9800",
    "missed": "#FF3B3B",
    "bola dentro": "#4CAF50"
}

# ======================================================
# UTILIDADES
# ======================================================

def cargar_datos(ruta=None):
    if ruta is None or ruta.strip() == "":
        ruta = os.path.join(os.path.dirname(__file__), "..", "data", "interim", "final_clean.parquet")
    ruta = os.path.abspath(ruta)
    print(f"📂 Cargando datos desde: {ruta}")

    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(ruta)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(ruta)
    else:
        raise ValueError(f"❌ Formato de archivo no soportado: {ext}")

    print(f"✅ Datos cargados: {len(df):,} filas.")
    return df

# ======================================================
# PROCESAMIENTO DE MARCADOR ROBUSTO
# ======================================================

def procesar_marcador_robusto(df_clean):
    df = df_clean.copy()
    cols = ["clip_start", "juego_p1", "juego_p2", "set_p1", "set_p2"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df = df.sort_values("clip_start").reset_index(drop=True)

    # === 1️⃣ numéricos ===
    for c in ["juego_p1", "juego_p2", "set_p1", "set_p2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["set_inferido_auto"] = False

    # === 2️⃣ propagar juegos ===
    df["juego_p1"] = df["juego_p1"].ffill().fillna(0).astype(int)
    df["juego_p2"] = df["juego_p2"].ffill().fillna(0).astype(int)

    # === 3️⃣ corregir sets SOLO si uno está NaN y el otro no ===
    for i in range(1, len(df)):
        s1_prev, s2_prev = df.loc[i - 1, ["set_p1", "set_p2"]]
        s1, s2 = df.loc[i, ["set_p1", "set_p2"]]

        if pd.isna(s1) and pd.isna(s2):
            df.loc[i, ["set_p1", "set_p2"]] = [s1_prev, s2_prev]
            continue

        if pd.isna(s1) and pd.notna(s2):
            if s2 in (0, 1):
                df.loc[i, "set_p1"] = 1 - int(s2)
                df.loc[i, "set_inferido_auto"] = True
            else:
                df.loc[i, "set_p1"] = s1_prev
        elif pd.isna(s2) and pd.notna(s1):
            if s1 in (0, 1):
                df.loc[i, "set_p2"] = 1 - int(s1)
                df.loc[i, "set_inferido_auto"] = True
            else:
                df.loc[i, "set_p2"] = s2_prev

    df[["set_p1", "set_p2"]] = df[["set_p1", "set_p2"]].ffill().fillna(0).astype(int)

    # === 4️⃣ detectar cambios de set ===
    df["juego_p1_prev"] = df["juego_p1"].shift(1).fillna(0)
    df["juego_p2_prev"] = df["juego_p2"].shift(1).fillna(0)
    df["set_p1_prev"] = df["set_p1"].shift(1).fillna(0)
    df["set_p2_prev"] = df["set_p2"].shift(1).fillna(0)

    cond_set_exp = (df["set_p1"] != df["set_p1_prev"]) | (df["set_p2"] != df["set_p2_prev"])
    cond_set_imp = ((df["juego_p1"] == 0) & (df["juego_p2"] == 0)) & (
        (df["juego_p1_prev"] > 0) | (df["juego_p2_prev"] > 0)
    )
    df["cambio_set"] = cond_set_exp | cond_set_imp

    # === 5️⃣ inferir marcador del set cerrado ===
    def inferir_y_validar(row):
        p1, p2 = int(row["juego_p1_prev"]), int(row["juego_p2_prev"])
        if p1 == p2 == 0:
            return None
        if p1 > p2:
            f1, f2 = max(6, p1 + 1), p2
        elif p2 > p1:
            f1, f2 = p1, max(6, p2 + 1)
        else:
            return None
        if (
            (f1 >= 6 or f2 >= 6)
            and (
                abs(f1 - f2) >= 2
                or (f1 == 7 and f2 in (5, 6))
                or (f2 == 7 and f1 in (5, 6))
            )
        ):
            return (f1, f2)
        return None

    df["set_inferido"] = df.apply(lambda r: inferir_y_validar(r) if r["cambio_set"] else None, axis=1)
    df["juegos_finalizados_inferidos"] = df["set_inferido"].apply(lambda x: sum(x) if isinstance(x, tuple) else 0)

    # === 6️⃣ acumulados ===
    df["juegos_acumulados_sets"] = df["juegos_finalizados_inferidos"].cumsum()
    df["progreso_set_actual"] = df["juego_p1"] + df["juego_p2"]
    df.loc[df["cambio_set"], "progreso_set_actual"] = 0

    # === 7️⃣ contador de juegos ===
    df["juegos_totales_acumulados"] = 0
    contador = 0
    for i in range(len(df)):
        if i > 0:
            j_prev = f"{df.loc[i-1, 'juego_p1']}-{df.loc[i-1, 'juego_p2']}"
            j_cur = f"{df.loc[i, 'juego_p1']}-{df.loc[i, 'juego_p2']}"
            s_prev = f"{df.loc[i-1, 'set_p1']}-{df.loc[i-1, 'set_p2']}"
            s_cur = f"{df.loc[i, 'set_p1']}-{df.loc[i, 'set_p2']}"
            if (j_cur != j_prev) and not (j_cur == "0-0" and s_cur != s_prev):
                contador += 1
        df.loc[i, "juegos_totales_acumulados"] = contador

    # === 8️⃣ resumen de sets (robusto y sin duplicar) ===
    resumen_sets = []
    set_counter = 0
    run_s1, run_s2 = 0, 0
    last_set_recorded_idx = -1
    last_clip = None

    for i, r in df.iterrows():
        if bool(r["cambio_set"]) and isinstance(r["set_inferido"], tuple):
            p1g, p2g = r["set_inferido"]
            ganador = "P1" if p1g > p2g else "P2"

            # Evitar duplicados (clip_start repetido o mismo marcador)
            if (
                (last_clip is None or r["clip_start"] != last_clip)
                and (not resumen_sets or f"{p1g}-{p2g}" != resumen_sets[-1]["Marcador_Set"])
            ):
                set_counter += 1
                if ganador == "P1":
                    run_s1 += 1
                else:
                    run_s2 += 1
                resumen_sets.append({
                    "Set": set_counter,
                    "clip_start": r["clip_start"],
                    "Marcador_Set": f"{p1g}-{p2g}",
                    "Ganador": ganador,
                    "Marcador_Sets_Total": f"{run_s1}-{run_s2}"
                })
                last_set_recorded_idx = i
                last_clip = r["clip_start"]

    # === 9️⃣ inferir último set si no hubo cambio_set final ===
    ultima_fila = df.iloc[-1]
    if not df["cambio_set"].iloc[-1]:
        j1, j2 = int(ultima_fila["juego_p1"]), int(ultima_fila["juego_p2"])
        if j1 != j2:
            ganador = "P1" if j1 > j2 else "P2"
            set_counter += 1
            if ganador == "P1":
                run_s1 += 1
            else:
                run_s2 += 1
            marcador_final_set = f"{j1+1}-{j2}" if ganador == "P1" else f"{j1}-{j2+1}"
            resumen_sets.append({
                "Set": set_counter,
                "clip_start": ultima_fila["clip_start"],
                "Marcador_Set": marcador_final_set,
                "Ganador": ganador,
                "Marcador_Sets_Total": f"{run_s1}-{run_s2}"
            })

    df_resumen = pd.DataFrame(resumen_sets)
    return df, df_resumen

# ======================================================
# RESTO DE FUNCIONES (sin cambios de lógica)
# ======================================================

def recortar_por_limite(df, limite_juegos, resumen_sets):
    mask = df["juegos_totales_acumulados"] >= limite_juegos
    if mask.any():
        idx = mask.idxmax()
        fila = df.loc[idx]
        resumen_prev = resumen_sets.loc[resumen_sets["clip_start"] <= fila["clip_start"]]
        marcadores_cerrados = resumen_prev["Marcador_Set"].tolist() if not resumen_prev.empty else []
        marcador_completo = " ".join(marcadores_cerrados + [f"{int(fila['juego_p1'])}-{int(fila['juego_p2'])}"])
    else:
        fila = df.iloc[-1]
        marcadores_cerrados = resumen_sets["Marcador_Set"].tolist() if not resumen_sets.empty else []
        marcador_completo = " ".join(marcadores_cerrados)
    return fila, marcador_completo

def clasificar_eventos(df):
    d = df.copy()
    for c in [COL_WINNER, COL_ERROR]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].astype(str).str.lower().str.strip()

    # --- AÑADE ESTO PARA DEPURACIÓN ---
    # ----------------------------------
    d["categoria"] = "bola dentro"
    d.loc[d[COL_ERROR].str.contains("error no forzado", na=False), "categoria"] = "error no forzado"
    d.loc[d[COL_ERROR].str.contains("missed", na=False), "categoria"] = "missed"
    d.loc[d[COL_WINNER].str.contains("winner", na=False), "categoria"] = "winner"
    return d

def resumen_metricas_por_jugador(df):
    conteos = df.groupby(["jugador", "categoria"]).size().unstack(fill_value=0)
    conteos["total"] = conteos.sum(axis=1)
    porcentajes = conteos.div(conteos["total"], axis=0).mul(100).round(1)
    return pd.concat([conteos, porcentajes.add_suffix("_%")], axis=1).reset_index()

def top_golpes_por_jugador(df, output_dir, top_n=5):
    if "golpe_q" not in df.columns:
        print("⚠️ No hay columna 'golpe_q' para top golpes.")
        return
    conteo = df.groupby(["jugador", "golpe_q"]).size().reset_index(name="conteo")
    top = conteo.sort_values(["jugador", "conteo"], ascending=[True, False]).groupby("jugador").head(top_n)
    os.makedirs(output_dir, exist_ok=True)
    for jug, g in top.groupby("jugador"):
        plt.figure(figsize=(6, 4))
        plt.barh(g["golpe_q"], g["conteo"], color="#2196F3")
        plt.title(f"Top {top_n} golpes — {jug}")
        plt.xlabel("Repeticiones")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_golpes_{jug}.png"), dpi=300)
        plt.close()

# ======================================================
# DETECCIÓN AUTOMÁTICA DE COLUMNAS DE COORDENADAS
# ======================================================

def detectar_columnas_coordenadas(df):
    """
    Detecta automáticamente los nombres correctos de columnas
    para inicio_x, inicio_y, fin_x, fin_y, incluso si hay typos como 'gople' o ':'.
    """
    posibles_nombres = {
    "inicio_x": ["inicio_gople:_x", "inicio_gople_x", "inicio_golpe:_x", "inicio_golpe_x"],
    "inicio_y": ["inicio_gople:_y", "inicio_gople_y", "inicio_golpe:_y", "inicio_golpe_y"],
    "fin_x":    ["fin_gople:_x", "fin_gople_x", "fin_golpe:_x", "fin_golpe_x"],
    "fin_y":    ["fin_gople:_y", "fin_gople_y", "fin_golpe:_y", "fin_golpe_y"]
}


    encontrados = {}
    for clave, lista in posibles_nombres.items():
        for nombre in lista:
            if nombre in df.columns:
                encontrados[clave] = nombre
                break
        if clave not in encontrados:
            encontrados[clave] = None  # si no se encuentra

    #print("🧭 Columnas detectadas automáticamente:")
    for k, v in encontrados.items():
        print(f"  {k}: {v}")

    return encontrados


def pintar_pista_interactiva(df, output_dir="outputs/html_pistas"):
    import os
    import plotly.graph_objects as go
    # Importamos las constantes globales necesarias (asumo que están definidas fuera de la función)
    # COL_JUGADOR, COL_INICIO_X, COL_INICIO_Y, COL_FIN_X, COL_FIN_Y, 
    # ANCHO_PISTA, LARGO_PISTA, COLORES_EVENTO, etc.

    # Asegurarse de que la carpeta exista
    os.makedirs(output_dir, exist_ok=True)
    print(f"🎾 Guardando pistas interactivas en: {os.path.abspath(output_dir)}")

    jugadores = df[COL_JUGADOR].dropna().unique().tolist()
    print(f"👥 Jugadores detectados: {jugadores}")

    for jugador in jugadores:
        # Usamos las constantes de columna que definiste en tu código original
        sub = df[df[COL_JUGADOR] == jugador].dropna(
            subset=[COL_INICIO_X, COL_INICIO_Y, COL_FIN_X, COL_FIN_Y]
        ).reset_index(drop=True)
        
        if sub.empty:
            print(f"⚠️ Sin coordenadas válidas para {jugador}, no se genera pista.")
            continue

        # Crear figura
        fig = go.Figure()

        # --- DIBUJO DE LA PISTA DE PÁDEL (Líneas 33-40 de tu original, más detalle) ---
        SERVICIO_Y_OFFSET = 60 
        CENTRO_X = ANCHO_PISTA / 2
        
        fig.add_shape(type="rect", x0=0, y0=0, x1=ANCHO_PISTA, y1=LARGO_PISTA, line=dict(color="white", width=3)) 
        fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2, x1=ANCHO_PISTA, y1=LARGO_PISTA/2, line=dict(color="white", width=4)) 
        fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2 - SERVICIO_Y_OFFSET, x1=ANCHO_PISTA, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET, line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=0, y0=LARGO_PISTA/2 + SERVICIO_Y_OFFSET, x1=ANCHO_PISTA, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET, line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=CENTRO_X, y0=LARGO_PISTA/2, x1=CENTRO_X, y1=LARGO_PISTA/2 - SERVICIO_Y_OFFSET, line=dict(color="white", width=2))
        fig.add_shape(type="line", x0=CENTRO_X, y0=LARGO_PISTA/2, x1=CENTRO_X, y1=LARGO_PISTA/2 + SERVICIO_Y_OFFSET, line=dict(color="white", width=2))

        # --- DIBUJO DE TRAYECTORIAS CON DOBLE TRAZA (Marcador de inicio y fin diferente) ---
        traces_to_add = []
        
        for cat, color in COLORES_EVENTO.items():
            df_cat = sub[sub["categoria"] == cat]
            if df_cat.empty: continue
            
            is_first_trace = True
            
            for i, r in df_cat.iterrows():
                
                # 1. Traza de la LÍNEA + MARCADOR DE INICIO (Círculo)
                traces_to_add.append(go.Scatter(
                    x=[r[COL_INICIO_X], r[COL_FIN_X]], # La línea va de inicio a fin
                    y=[r[COL_INICIO_Y], r[COL_FIN_Y]], 
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    
                    # --- MARCADOR DE INICIO: Círculo ---
                    marker=dict(size=5, symbol='circle'), 
                    
                    showlegend=is_first_trace, 
                    name=cat,
                    hovertext=f"Golpe ({cat})",
                    hoverinfo="text",
                    legendgroup=cat, # Mismo grupo
                    
                    # Ocultación a prueba de fallos
                    unselected=dict(marker=dict(opacity=0)) 
                ))

                # 2. Traza de solo el MARCADOR FINAL (Triángulo/Flecha)
                traces_to_add.append(go.Scatter(
                    x=[r[COL_FIN_X]], # Solo el punto final
                    y=[r[COL_FIN_Y]],
                    mode="markers", # SOLO marcadores
                    
                    # --- MARCADOR DE FIN: Triángulo/Flecha ---
                    marker=dict(size=8, symbol='triangle-right', color=color),
                    
                    showlegend=False, # Nunca muestra esta traza en la leyenda
                    name=cat,
                    hovertext=f"Fin ({cat})",
                    hoverinfo="text",
                    legendgroup=cat, # Mismo grupo
                    
                    # Ocultación a prueba de fallos
                    unselected=dict(marker=dict(opacity=0))
                ))
                
                is_first_trace = False

        fig.add_traces(traces_to_add)

        # --- CONFIGURACIÓN VISUAL, TAMAÑO Y LEYENDA (Tu original modificado) ---
        fig.update_layout(
            title=f"Pista Interactivo — {jugador} (Inicio/Fin Diferenciado)",
            plot_bgcolor="#003C77",
            paper_bgcolor="#00244D",
            xaxis=dict(visible=False, range=[0, ANCHO_PISTA]),
            # Aseguramos el ratio 1:1 y el rango de Y correcto
            yaxis=dict(visible=False, range=[0,LARGO_PISTA], scaleanchor="x", scaleratio=1), 
            font=dict(color="white"),
            showlegend=True,
            
            # --- CONFIGURACIÓN PARA OCULTACIÓN TOTAL AL CLIC ---
            legend=dict(
                groupclick="togglegroup", 
                itemclick="toggle"      
            )
            # ----------------------------------------------------
        )

        # --- RUTA DE GUARDADO (Tu original) ---
        salida_html = os.path.join(output_dir, f"pista_{jugador}_diferenciada.html")
        fig.write_html(salida_html)
        print(f"✅ Archivo generado: {os.path.abspath(salida_html)}")

    print("✅ Finalizado: pistas interactivas creadas correctamente.")


# ======================================================
# DIRECTORIO DE SALIDA
# ======================================================

def build_output_dir(base_dir, nombre_partido, marcador_texto):
    os.makedirs(base_dir, exist_ok=True)
    marcador_texto = str(marcador_texto).replace(" ", "_")
    path = os.path.join(base_dir, nombre_partido, marcador_texto)
    os.makedirs(path, exist_ok=True)
    return path


# ======================================================
# PIPELINE PRINCIPAL
# ======================================================

def analizar_partido_interactivo():
    # === 1️⃣ Cargar datos ===
    ruta = input("📂 Ruta del archivo (Excel o Parquet): ").strip()
    df0 = cargar_datos(ruta)
    n_juegos = int(input("🎮 ¿Cuántos juegos quieres analizar?: ").strip())

    # === 2️⃣ Detectar nombre del partido ===
    ultima_col = df0.columns[-1]
    valor_col = str(df0[ultima_col].iloc[0]).strip()
    if valor_col and valor_col.lower() not in ("nan", "", "none"):
        nombre_partido = valor_col
    else:
        nombre_partido = os.path.splitext(os.path.basename(ruta))[0]
    print(f"🏷️ Nombre del partido detectado: {nombre_partido}")

    # === 3️⃣ Procesar marcador robusto sobre el DataFrame completo ===
    df_proc, df_resumen = procesar_marcador_robusto(df0)

    # === 4️⃣ Determinar fila de corte y marcador actual ===
    fila, marcador_completo = recortar_por_limite(df_proc, n_juegos, df_resumen)
    print(f"Este es el marcador completo hasta el corte: {marcador_completo}")
    #print(f"🎯 Marcador parcial — Juegos: {fila['juego_p1']}-{fila['juego_p2']} | Sets: {fila['set_p1']}-{fila['set_p2']}")

    # === 5️⃣ Cortar el DataFrame original hasta esa fila ===
    if "clip_start" in df0.columns:
        # Buscar el índice más cercano del clip_start coincidente o menor
        idx_corte = df0[df0["clip_start"] <= fila["clip_start"]].index.max()
        df_cortado = df0.loc[:idx_corte].copy()
    else:
        df_cortado = df0.copy()  # Fallback
    print(f"✂️  DataFrame recortado hasta fila {idx_corte} ({len(df_cortado)} filas).")

    # === 6️⃣ Crear carpeta de salida ===
    base = os.path.join("outputs", "figures")
    out_dir = build_output_dir(base, nombre_partido, marcador_completo)
    print(f"📁 Resultados guardados en: {os.path.abspath(out_dir)}")

    # === 7️⃣ Clasificación y métricas usando solo el DF recortado ===
    df_rec = clasificar_eventos(df_cortado)

    # Validar que tenga columna 'jugador'
    if "jugador" not in df_rec.columns:
        print("⚠️ Advertencia: No se encontró la columna 'jugador' en los datos. Saltando métricas por jugador.")
    else:
        resumen = resumen_metricas_por_jugador(df_rec)
        resumen.to_csv(os.path.join(out_dir, "resumen_metricas.csv"), index=False)
        print(f"✅ Resumen guardado en {out_dir}")

        # === 8️⃣ Visualizaciones ===
        top_golpes_por_jugador(df_rec, output_dir=out_dir)
        # === Detectar columnas de coordenadas automáticamente ===
        col_map = detectar_columnas_coordenadas(df_rec)

        # Actualizar variables globales
        global COL_INICIO_X, COL_INICIO_Y, COL_FIN_X, COL_FIN_Y
        if col_map["inicio_x"]: COL_INICIO_X = col_map["inicio_x"]
        if col_map["inicio_y"]: COL_INICIO_Y = col_map["inicio_y"]
        if col_map["fin_x"]:    COL_FIN_X    = col_map["fin_x"]
        if col_map["fin_y"]:    COL_FIN_Y    = col_map["fin_y"]

        #print(f"✅ Usando columnas: {COL_INICIO_X}, {COL_INICIO_Y}, {COL_FIN_X}, {COL_FIN_Y}")

        pintar_pista_interactiva(df_rec, output_dir=out_dir)

    print("\n✅ Análisis completo.")
    return df_cortado, marcador_completo, out_dir


# ======================================================
# EJECUCIÓN
# ======================================================

if __name__ == "__main__":
    analizar_partido_interactivo()
