import logging
import sys
from pathlib import Path

# Resolve absolute path to the project root (assumes config.py is in src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define main data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Define explicit file destinations
CUSTOMERS_FILE = RAW_DIR / "customers.parquet"
PRODUCTS_FILE = RAW_DIR / "products.parquet"
ORDERS_FILE = RAW_DIR / "orders.parquet"

def setup_logging(name: str) -> logging.Logger:
    """
    Sets up and returns a configured logger for standard stdout output.

    Args:
        name (str): The name of the module requesting the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Ensure directories exist upon import
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)