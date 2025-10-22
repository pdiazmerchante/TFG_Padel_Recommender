"""
Verifica que los módulos principales de src/ se pueden importar correctamente.
"""

import sys
from pathlib import Path

# Forzar que Python vea la raíz del proyecto
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print("=== Verificación de imports ===")

try:
    import src
    print("OK: módulo raíz 'src' encontrado.")
except ImportError as e:
    print("❌ ERROR: no se puede importar 'src' ->", e)

try:
    from src.common.logging_setup import setup_logging
    print("OK: módulo 'src.common.logging_setup' encontrado.")
except ImportError as e:
    print("❌ ERROR: no se puede importar 'src.common.logging_setup' ->", e)

try:
    from src.data.load_data import load_raw_data
    print("OK: módulo 'src.data.load_data' encontrado.")
except ImportError as e:
    print("❌ ERROR: no se puede importar 'src.data.load_data' ->", e)

print("=== Fin de la verificación ===")
