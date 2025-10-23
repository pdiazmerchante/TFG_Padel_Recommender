"""
Pipeline completo de limpieza y validación de datos de pádel.
------------------------------------------------------------
1️⃣ Elimina archivos intermedios antiguos (seguridad)
2️⃣ Lee CSV(s) desde data/raw/
3️⃣ Colapsa filas de un mismo evento (sin limpiar antes)
4️⃣ Limpia datos resultantes del colapso (minúsculas, tipos, normalización)
5️⃣ Valida estructura y tipos sobre df_clean
6️⃣ Guarda resultados intermedios y finales
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
from src.common.logging_setup import setup_logging
from src.data.load_data import load_raw_data, load_config
from src.data.event_collapse import collapse_events
from src.data.validate_raw import validate_raw
from src.data.clean_data import clean_dataset
from src.data.schemas import RAW_SCHEMA  # si tienes un esquema específico para 'clean', cámbialo aquí


def main():
    logger = setup_logging()
    logger.info("🚀 Iniciando pipeline completo de limpieza")

    # --- 0️⃣ LIMPIAR INTERMEDIOS ANTIGUOS ---
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path("data/metadata")
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

    logger.info("🧹 Limpieza previa: eliminados archivos intermedios antiguos.")

    # --- 1️⃣ LEER CSV(s) ---
    _ = load_config("config/config.toml")  # mantener si tu loader necesita side-effects
    df_raw = load_raw_data("config/config.toml")
    logger.info(f"RAW: {len(df_raw):,} filas, {len(df_raw.columns)} columnas")

    # --- 2️⃣ COLAPSAR EVENTOS (sin limpiar antes) ---
    try:
        df_collapsed = collapse_events(df_raw)
        logger.info(f"COLLAPSED: {len(df_collapsed):,} filas, {len(df_collapsed.columns)} columnas")
    except Exception as e:
        logger.error(f"❌ Error al colapsar eventos: {e}")
        return

    # --- 3️⃣ LIMPIEZA (minúsculas, tipos, normalización) SOBRE COLLAPSED ---
    try:
        df_clean = clean_dataset(df_collapsed)
        logger.info(f"CLEAN: {len(df_clean):,} filas, {len(df_clean.columns)} columnas")
    except Exception as e:
        logger.error(f"❌ Error durante la limpieza: {e}")
        return

    # --- 4️⃣ VALIDACIÓN (sobre df_clean) ---
    try:
        report = validate_raw(df_clean, RAW_SCHEMA)  # usa el esquema apropiado si tienes uno para 'clean'
        with open(metadata_dir / "quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("🧾 Reporte de calidad guardado en data/metadata/quality_report.json")
    except Exception as e:
        logger.warning(f"⚠️ Validación falló parcialmente: {e}")

    # --- 5️⃣ GUARDAR RESULTADOS ---
    out_raw = interim_dir / "raw_concat.parquet"             # salida directa del merge de CSVs
    out_collapsed = interim_dir / "events_collapsed.parquet" # tras collapse (sin limpiar)
    out_clean = interim_dir / "final_clean.parquet"          # tras cleaning (minúsculas, tipos, etc.)

    # Guardamos cada etapa claramente
    try:
        # Nota: df_raw puede ser muy grande; si prefieres no guardar, comenta esta línea
        df_raw.to_parquet(out_raw, index=False)
        df_collapsed.to_parquet(out_collapsed, index=False)
        df_clean.to_parquet(out_clean, index=False)
    except Exception as e:
        logger.error(f"❌ Error al guardar salidas: {e}")
        return

    logger.info("✅ Guardados:")
    logger.info(f"   → {out_raw}")
    logger.info(f"   → {out_collapsed}")
    logger.info(f"   → {out_clean}")
    logger.info("🎯 Pipeline completo terminado correctamente.")


if __name__ == "__main__":
    main()
