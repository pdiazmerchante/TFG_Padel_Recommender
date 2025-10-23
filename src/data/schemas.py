"""
Definición del esquema de validación para los datos M3.
Este esquema sirve para verificar que las columnas mínimas existen
y que los tipos son razonables.
"""

from src.data.validate_raw import RawSchema

RAW_SCHEMA = RawSchema(
    required_cols=[
        # columnas que deben existir sí o sí
        "row_name", "punto_p1", "punto_p2", "error"
    ],
    warn_if_missing=[
        # columnas deseables pero no críticas
        "set", "game", "lado", "inicio_golpe_x", "inicio_golpe_y"
    ],
    numeric_should_be=[
        # columnas que deberían ser numéricas
        "punto_p1", "punto_p2", "juego_p1", "juego_p2"
    ],
    string_should_be=[
        # columnas textuales
        "jugador", "pareja", "cara_pala"
    ],
    max_null_frac=0.4
)
