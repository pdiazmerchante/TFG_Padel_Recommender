import pandas as pd

PUNTOS_MAP = {
    "0": "0",
    "15": "15",
    "30": "30",
    "40": "40",
    "Adv": "adv"
}

def crear_marcador(df):
    df = df.copy()

    # Orden cronológico
    df = df.sort_values("clip_start").reset_index(drop=True)

    # ----------------------------
    # 1️⃣ SETS y JUEGOS: NUMÉRICOS + PROPAGACIÓN
    # ----------------------------
    for c in ["set_p1", "set_p2", "juego_p1", "juego_p2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].ffill().fillna(0).astype(int)

    # ----------------------------
    # 2️⃣ PUNTOS: NO CONVERTIR A NUMÉRICO → Propagar como strings
    # ----------------------------
    for c in ["punto_p1", "punto_p2"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

            # Reemplazar NAs por pandas.NA
            df[c] = df[c].replace(["nan", "None", ""], pd.NA)

            # Propagar valor previo
            df[c] = df[c].ffill()

            # Los NA iniciales: poner 0
            df[c] = df[c].fillna("0")

    # ----------------------------
    # 3️⃣ Convertir puntuaciones a formato correcto (0/15/30/40/AD)
    # ----------------------------
    df["punto_p1"] = df["punto_p1"].apply(lambda x: PUNTOS_MAP.get(str(x), "0"))
    df["punto_p2"] = df["punto_p2"].apply(lambda x: PUNTOS_MAP.get(str(x), "0"))

    # ----------------------------
    # 4️⃣ Construir columnas de marcador
    # ----------------------------
    df["marcador_sets"] = df["set_p1"].astype(str) + "-" + df["set_p2"].astype(str)
    df["marcador_juegos"] = df["juego_p1"].astype(str) + "-" + df["juego_p2"].astype(str)
    df["marcador_puntos"] = df["punto_p1"] + "-" + df["punto_p2"]

    df["marcador"] = (
        df["marcador_sets"] + " | " +
        df["marcador_juegos"] + " | " +
        df["marcador_puntos"]
    )

    return df


def asignar_informacion_saque_y_punto(df):
    df = df.copy()

    # -----------------------------------------
    # 1️⃣ DETERMINAR PAREJA 1 Y PAREJA 2
    # -----------------------------------------
    primeras_parejas = df["pareja"].dropna().unique()
    if len(primeras_parejas) < 2:
        print("⚠ No se detectaron dos parejas distintas.")
        return df

    pareja1 = primeras_parejas[0]
    pareja2 = primeras_parejas[1]

    jugadores_p1 = set(pareja1.split("-"))
    jugadores_p2 = set(pareja2.split("-"))

    def obtener_pareja(jugador):
        if jugador in jugadores_p1:
            return 1
        if jugador in jugadores_p2:
            return 2
        return None

    df["pareja_jugador"] = df["jugador"].apply(obtener_pareja)

    # -----------------------------------------
    # 2️⃣ EXTRAER NOMBRE DEL SACADOR
    # -----------------------------------------
    def extraer_sacador(txt):
        if pd.isna(txt):
            return None
        # formato: "1º Servicio Galán"
        parts = str(txt).split()
        return parts[-1] if len(parts) >= 3 else None

    df["sacador"] = df["servicio"].apply(extraer_sacador)
    df["pareja_sacador"] = df["sacador"].apply(obtener_pareja)

    # -----------------------------------------
    # 3️⃣ IDENTIFICAR QUIÉN GANA EL PUNTO
    # -----------------------------------------
    # Comparamos con la fila anterior
    df["punto_p1_prev"] = df["punto_p1"].shift(1)
    df["punto_p2_prev"] = df["punto_p2"].shift(1)

    def ganador_punto(row):
        if pd.isna(row["punto_p1_prev"]) or pd.isna(row["punto_p2_prev"]):
            return None
        if row["punto_p1"] != row["punto_p1_prev"]:
            return 1
        if row["punto_p2"] != row["punto_p2_prev"]:
            return 2
        return None

    df["pareja_ganadora_punto"] = df.apply(ganador_punto, axis=1)

    # -----------------------------------------
    # 4️⃣ SACADOR GANA EL PUNTO?
    # -----------------------------------------
    def gana_saque(row):
        if row["pareja_sacador"] is None or row["pareja_ganadora_punto"] is None:
            return None
        return int(row["pareja_sacador"] == row["pareja_ganadora_punto"])

    df["gana_punto_sacador"] = df.apply(gana_saque, axis=1)

    return df
