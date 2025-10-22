"""
Script de ingesta inicial: lee los CSV desde data/raw, los une y los guarda en data/interim/.
Funciona tanto si se ejecuta con `python scripts/ingest_raw.py` como con `python -m scripts.ingest_raw`.
"""

import sys
from pathlib import Path

# --- FORZAR AÑADIR LA RAÍZ DEL PROYECTO AL PATH (para que Python vea src/) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- IMPORTS DEL PROYECTO ---
from src.common.logging_setup import setup_logging
from src.data.load_data import load_raw_data, load_config


def main():
    logger = setup_logging()
    cfg = load_config("config/config.toml")

    interim_dir = Path(cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(config_path="config/config.toml")
    out_path = interim_dir / "raw_concat.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"✅ Datos combinados guardados en {out_path}")


if __name__ == "__main__":
    main()
