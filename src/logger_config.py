"""
Created on Sep, 2025
Author: s Bostan

Description:
    Set up logging for AI Search Optimizer, with console and rotating file handlers.

Licensed under the Apache License 2.0.
"""

import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs folder if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "ai_search.log")

# Create logger
logger = logging.getLogger("AI_Search_Logger")
logger.setLevel(logging.INFO)  # default level, can be changed to DEBUG

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

# File handler with rotation (5 MB max, keep 3 files)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)