"""
Pipeline completo de limpieza y validación de datos de pádel.
------------------------------------------------------------
1️⃣ Elimina archivos intermedios antiguos
2️⃣ Lee CSV(s) desde data/raw/
3️⃣ Colapsa filas del mismo evento
4️⃣ Normaliza columnas duplicadas/incorrectas
5️⃣ Limpia datos (tipos, texto, normalización)
6️⃣ Valida estructura
7️⃣ Genera datasets procesados finales
"""

import sys
from pathlib import Path
import pandas as pd
import json

# === Acceso a src/ ===
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# === Imports de tus módulos ===
from src.data.normalize_columns import normalizar_columnas
from src.common.logging_setup import setup_logging
from src.data.load_data import load_raw_data, load_config
from src.data.event_collapse import collapse_events
from src.data.validate_raw import validate_raw
from src.data.clean_data import clean_dataset
from src.data.schemas import RAW_SCHEMA
from src.data.score_utils import crear_marcador
from src.data.score_utils import asignar_informacion_saque_y_punto


def main():
    logger = setup_logging()
    logger.info("🚀 Iniciando pipeline completo de limpieza")

    # ============================================================
    # 0️⃣ LIMPIEZA PREVIA
    # ============================================================
    interim_dir = Path("data/interim")
    metadata_dir = Path("data/metadata")
    interim_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for f in interim_dir.glob("*.parquet"):
        try:
            f.unlink()
        except Exception as e:
            logger.warning(f"No se pudo borrar {f}: {e}")

    for f in metadata_dir.glob("*.json"):
        try:
            f.unlink()
        except Exception as e:
            logger.warning(f"No se pudo borrar {f}: {e}")

    logger.info("🧹 Limpieza previa realizada.")

    # ============================================================
    # 1️⃣ CARGA RAW
    # ============================================================
    _ = load_config("config/config.toml")
    df_raw = load_raw_data("config/config.toml")
    logger.info(f"RAW: {len(df_raw):,} filas, {len(df_raw.columns)} columnas")

    # ============================================================
    # 2️⃣ COLAPSAR EVENTOS
    # ============================================================
    try:
        df_collapsed = collapse_events(df_raw)
        logger.info(f"COLLAPSED: {len(df_collapsed):,} filas, {len(df_collapsed.columns)} columnas")
    except Exception as e:
        logger.error(f"❌ Error al colapsar eventos: {e}")
        return

    # ============================================================
    # 3️⃣ NORMALIZACIÓN DE COLUMNAS
    # ============================================================
    try:
        df_norm = normalizar_columnas(df_collapsed)
        logger.info(f"NORMALIZED: {len(df_norm):,} filas, {len(df_norm.columns)} columnas")
    except Exception as e:
        logger.error(f"❌ Error normalizando columnas: {e}")
        return

    # ============================================================
    # 4️⃣ LIMPIEZA GENERAL + MARCADOR
    # ============================================================
    try:
        df_clean = clean_dataset(df_norm)
        df_clean = crear_marcador(df_clean)
        df_clean = asignar_informacion_saque_y_punto(df_clean)

        logger.info("MARCADOR añadido correctamente.")
        logger.info(f"CLEAN: {len(df_clean):,} filas, {len(df_clean.columns)} columnas")
    except Exception as e:
        logger.error(f"❌ Error durante limpieza: {e}")
        return

    # ============================================================
    # 5️⃣ VALIDACIÓN
    # ============================================================
    try:
        report = validate_raw(df_clean, RAW_SCHEMA)
        with open(metadata_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("🧾 Reporte de calidad guardado.")
    except Exception as e:
        logger.warning(f"⚠ Validación parcial: {e}")

    # ============================================================
    # 6️⃣ GUARDAR INTERMEDIOS
    # ============================================================
    out_raw = interim_dir / "raw_concat.parquet"
    out_collapsed = interim_dir / "events_collapsed.parquet"
    out_clean = interim_dir / "final_clean.parquet"

    try:
        df_raw.to_parquet(out_raw, index=False)
        df_collapsed.to_parquet(out_collapsed, index=False)
        df_clean.to_parquet(out_clean, index=False)
    except Exception as e:
        logger.error(f"❌ Error guardando intermedios: {e}")
        return

    # ============================================================
    # 7️⃣ DATASETS PROCESADOS
    # ============================================================
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_clean = df_clean.sort_values("clip_start").reset_index(drop=True)

    # ---------------------------
    # SETS
    # ---------------------------
    if "set_num" in df_clean.columns:
        df_sets = df_clean.groupby("set_num").agg("first").reset_index()
        df_sets.to_parquet(processed_dir / "sets.parquet", index=False)

    # ---------------------------
    # JUEGOS
    # ---------------------------
    if {"juego_p1", "juego_p2"}.issubset(df_clean.columns):
        df_juegos = df_clean.groupby(["set_num", "juego_p1", "juego_p2"]).agg("first").reset_index()
        df_juegos.to_parquet(processed_dir / "juegos.parquet", index=False)

    # ---------------------------
    # PUNTOS
    # ---------------------------
    cols_puntos = [c for c in ["punto_ganado", "punto_perdido", "winner"] if c in df_clean.columns]
    if cols_puntos:
        df_puntos = df_clean.dropna(subset=cols_puntos, how="all")
        df_puntos.to_parquet(processed_dir / "puntos.parquet", index=False)

    # ============================================================
    # 🔥 GOLPES (TU DATASET PRINCIPAL)
    # ============================================================
    if "golpe_q" in df_clean.columns:

        df_golpes = df_clean[df_clean["golpe_q"].notna() & (df_clean["golpe_q"] != "")]
        df_golpes = df_golpes.sort_values("clip_start")

        # -------------------------
        # CATEGORIA PUNTO
        # -------------------------
        def obtener_categoria(row):
            for col in ["error", "winner", "fuerza_error"]:
                val = row.get(col)
                if pd.notna(val) and str(val).strip() != "":
                    return val
            return "bola dentro"

        df_golpes["categoria_punto"] = df_golpes.apply(obtener_categoria, axis=1)

        # ============================================================
        # 🔒 NORMALIZACIÓN DEFINITIVA DE COORDENADAS
        # ============================================================
        COORD_ALIASES = {
            "inicio_golpe_x": "inicio_x",
            "inicio_golpe_y": "inicio_y",
            "inicio_gople_x": "inicio_x",
            "inicio_gople_y": "inicio_y",
            "inicio_golpe:_x": "inicio_x",
            "inicio_golpe:_y": "inicio_y",
            "inicio_gople:_x": "inicio_x",
            "inicio_gople:_y": "inicio_y",

            "fin_golpe_x": "fin_x",
            "fin_golpe_y": "fin_y",
            "fin_golpe:_x": "fin_x",
            "fin_golpe:_y": "fin_y",

            "start_x": "inicio_x",
            "start_y": "inicio_y",
            "end_x": "fin_x",
            "end_y": "fin_y",
        }

        df_golpes = df_golpes.rename(columns={c: COORD_ALIASES[c] for c in df_golpes.columns if c in COORD_ALIASES})

        # siempre crear columnas aunque estén vacías
        for col in ["inicio_x", "inicio_y", "fin_x", "fin_y"]:
            if col not in df_golpes.columns:
                df_golpes[col] = pd.NA

        # ============================================================
        # 🔧 LIMPIEZA SEGURA FINAL
        # ============================================================

        # columnas a eliminar (time y rarezas)
        columnas_prohibidas = [
            c for c in df_golpes.columns
            if (
                ("time" in c.lower()) or 
                c.endswith(":_x") or 
                c.endswith(":_y")
            )
            and c not in ["inicio_x", "inicio_y", "fin_x", "fin_y"]
        ]

        df_golpes = df_golpes.drop(columns=columnas_prohibidas, errors="ignore")

        # -------- FIX DE TU ERROR --------
        cols_empty = [
            c for c in df_golpes.columns
            if (c not in ["inicio_x", "inicio_y", "fin_x", "fin_y"])
            and df_golpes[c].isna().all()
        ]
        # ----------------------------------

        df_golpes = df_golpes.drop(columns=cols_empty, errors="ignore")

        df_golpes = df_golpes.loc[:, ~df_golpes.columns.duplicated()]

        # ordenar
        columnas_principales = [
            "golpe_q", "cara_pala", "marcador", "jugador",
            "categoria_punto", "error", "winner", "fuerza_error",
            "zona_saque", "zona_resto",
            "inicio_x", "inicio_y", "fin_x", "fin_y"
        ]

        columnas_ordenadas = [c for c in columnas_principales if c in df_golpes.columns] + \
                             [c for c in df_golpes.columns if c not in columnas_principales]

        df_golpes = df_golpes[columnas_ordenadas]

        # guardar
        df_golpes.to_parquet(processed_dir / "golpes.parquet", index=False)

    logger.info("📦 Datasets procesados guardados en data/processed/")
    logger.info("🎯 Pipeline completo terminado correctamente.")


if __name__ == "__main__":
    main()
