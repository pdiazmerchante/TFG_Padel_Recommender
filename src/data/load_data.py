from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd
import logging

# Python 3.11 ya incluye tomllib, si usas 3.10 instala tomli
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

logger = logging.getLogger(__name__)

def load_config(config_path: str | Path = "config/config.toml") -> dict:
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg

def list_raw_files(raw_dir: str | Path, pattern: str) -> list[Path]:
    raw_dir = Path(raw_dir)
    return sorted(Path(p) for p in glob.glob(str(raw_dir / pattern)))

def read_single_csv(path: Path, sep: str, encoding: str, opts: dict, add_source: bool) -> pd.DataFrame:
    logger.info(f"Leyendo: {path.name}")
    df = pd.read_csv(path, sep=sep, encoding=encoding, **opts)
    if add_source:
        df["__source_file"] = path.name
    return df

def load_raw_data(config_path: str | Path = "config/config.toml", only_file: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    pattern = cfg["data"]["glob_pattern"]
    sep = cfg["data"]["separator"]
    encoding = cfg["data"]["encoding"]
    add_source = cfg["data"]["add_source_column"]
    opts = cfg.get("read_csv", {})

    if only_file:
        paths = [raw_dir / only_file]
    else:
        paths = list_raw_files(raw_dir, pattern)

    if not paths:
        raise FileNotFoundError(f"No hay CSV en {raw_dir}")

    dfs = [read_single_csv(p, sep, encoding, opts, add_source) for p in paths]
    df_all = pd.concat(dfs, ignore_index=True, sort=False)

    if cfg["data"]["infer_dtypes"]:
        df_all = df_all.convert_dtypes()

    logger.info(f"Total filas: {len(df_all):,} | Columnas: {len(df_all.columns)}")
    return df_all
