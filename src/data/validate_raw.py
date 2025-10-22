"""
Validación mínima del dataset RAW:
- Verifica columnas obligatorias
- Chequea % de nulos por columna
- Comprueba tipos básicos (numérico / string) cuando aplica
- Devuelve un dict con el "informe de calidad" y raise si falta algo crítico
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

@dataclass
class RawSchema:
    required_cols: List[str]           # columnas que deben existir
    warn_if_missing: List[str]         # no críticas: log warning si faltan
    numeric_should_be: List[str]       # columnas que esperas numéricas
    string_should_be: List[str]        # columnas que esperas string-like
    max_null_frac: float = 0.4         # umbral de nulos tolerable por columna

def validate_raw(df: pd.DataFrame, schema: RawSchema) -> Dict[str, Any]:
    report: Dict[str, Any] = {"missing_required": [], "missing_warn": [], "null_frac": {}, "type_warnings": []}

    # 1) columnas
    for c in schema.required_cols:
        if c not in df.columns:
            report["missing_required"].append(c)
    for c in schema.warn_if_missing:
        if c not in df.columns:
            report["missing_warn"].append(c)

    if report["missing_required"]:
        raise ValueError(f"Faltan columnas obligatorias: {report['missing_required']}")

    # 2) nulos (%)
    null_frac = df.isna().mean().to_dict()
    report["null_frac"] = {k: float(v) for k, v in null_frac.items()}

    # 3) tipos esperados (suave: avisamos si no coincide)
    for c in schema.numeric_should_be:
        if c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                report["type_warnings"].append(f"Esperado numérico en '{c}', pero dtype={df[c].dtype}")

    for c in schema.string_should_be:
        if c in df.columns:
            if not pd.api.types.is_string_dtype(df[c]):
                report["type_warnings"].append(f"Esperado string en '{c}', pero dtype={df[c].dtype}")

    # 4) umbral de nulos duro: si alguna columna supera max_null_frac -> warning (no paramos)
    report["high_nulls"] = [c for c, f in report["null_frac"].items() if f > schema.max_null_frac]

    return report
