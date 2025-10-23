# scripts/collapse_events.py
"""
Colapsa filas del mismo evento (Row Name, Clip Start, Clip End) en una sola fila
usando 'primer valor no nulo por columna'.

Uso:
    python scripts/collapse_events.py
o:
    python -m scripts.collapse_events
"""

import sys
from pathlib import Path

# Asegura acceso a src/ al ejecutar como archivo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.common.logging_setup import setup_logging
from src.data.event_collapse import collapse_events
import pandas as pd

def main():
    logger = setup_logging()

    in_path = Path("data/interim/raw_concat.parquet")
    if not in_path.exists():
        logger.error("❌ No se encuentra data/interim/raw_concat.parquet. Ejecuta antes scripts/ingest_raw.py")
        return

    logger.info(f"📥 Leyendo {in_path}")
    df = pd.read_parquet(in_path)

    logger.info("🔧 Colapsando eventos por (Row Name, Clip Start, Clip End) → primer valor no nulo por columna")
    df_out = collapse_events(df)

    out_path = Path("data/interim/events_collapsed.parquet")
    df_out.to_parquet(out_path, index=False)
    logger.info(f"✅ Guardado: {out_path}  | filas={len(df_out):,}, cols={len(df_out.columns)}")

if __name__ == "__main__":
    main()
