# src/data/event_collapse.py
from __future__ import annotations
from typing import List, Callable, Dict
import pandas as pd

def _first_non_null(s: pd.Series):
    """Devuelve el primer valor no nulo de la serie, o NA si todo es nulo."""
    # Usamos notna() para respetar pandas.NA, None y NaN
    m = s.notna()
    return s[m].iloc[0] if m.any() else pd.NA

def resolve_keys(df: pd.DataFrame, candidate_sets: List[List[str]]) -> List[str]:
    """
    Dado un DataFrame y una lista de listas de posibles claves,
    devuelve la primera combinación que exista completa en el df.
    """
    for keys in candidate_sets:
        if all(k in df.columns for k in keys):
            return keys
    raise KeyError(
        "No se encuentran las columnas de clave para agrupar. "
        "Probé con: " + " | ".join([", ".join(k) for k in candidate_sets])
    )

def collapse_events(
    df: pd.DataFrame,
    keys: List[str] | None = None,
    reducer: Callable[[pd.Series], object] = _first_non_null,
) -> pd.DataFrame:
    """
    Colapsa múltiples filas del mismo evento a una sola fila,
    tomando el primer valor no nulo por columna.

    - keys: columnas que identifican un evento. Si None, intenta:
        ["Row Name","Clip Start","Clip End"]  (nombres originales)
        ["row_name","clip_start","clip_end"]  (snake_case)
    - reducer: función de agregación por defecto = primer no nulo.
    """
    if keys is None:
        keys = resolve_keys(df, [
            ["Row Name", "Clip Start", "Clip End"],
            ["row_name", "clip_start", "clip_end"],
        ])

    # Construimos un dict de agregación: cada columna -> reducer, excepto las keys
    agg_map: Dict[str, Callable[[pd.Series], object]] = {
        c: reducer for c in df.columns if c not in keys
    }

    # groupby + agregación. Mantener orden de keys como columnas “normales”
    df_out = (
        df.groupby(keys, as_index=False)
          .agg(agg_map)
    )

    return df_out
