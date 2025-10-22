import logging

def setup_logging(level=logging.INFO):
    """
    Configura logging básico para mostrar mensajes con hora y nivel.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    return logging.getLogger("tfg")
