import logging
import os
from pathlib import Path

def configure_logging(log_path=None):
    """Configure a named application logger; tests can pass a temporary path."""
    logger = logging.getLogger("business_prediction")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers when the app is reloaded in development.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    destination = log_path or os.getenv("APP_LOG_PATH", "logs/app.log")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
