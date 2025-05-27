import logging
from pathlib import Path
import os
import sys
import torch
import json # Added import

# Load configuration at the start of the script
CONFIG = {}
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    # Use a basic print here since logger isn't set up yet
    print("WARNING: config.json not found in logging_config.py. Using default log path logic.")
except json.JSONDecodeError:
    print("WARNING: Error decoding config.json in logging_config.py. Using default log path logic.")

def setup_logging():
    """Set up logging configuration."""
    # Determine log directory: Env Var > Config File > Default
    env_log_dir = os.environ.get('LOG_DIR')
    config_log_dir = CONFIG.get('log_dir')

    if env_log_dir:
        log_dir_path_str = env_log_dir
        print(f"Logging_config: Using log directory from LOG_DIR environment variable: {log_dir_path_str}")
    elif config_log_dir and isinstance(config_log_dir, str) and config_log_dir.strip():
        log_dir_path_str = config_log_dir
        print(f"Logging_config: Using log directory from config.json: {log_dir_path_str}")
    else:
        if config_log_dir: # Log if it was present but invalid
             print(f"Logging_config: log_dir in config.json is present but empty or invalid: '{config_log_dir}'. Using default 'logs'.")
        log_dir_path_str = 'logs' # Default directory
        print(f"Logging_config: Using default log directory: {log_dir_path_str}")

    log_dir = Path(log_dir_path_str)
    log_dir.mkdir(exist_ok=True)
    
    # Use a single log file that gets overwritten
    log_file = log_dir / 'training.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Set to INFO by default
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            # File handler - overwrites the file on each run
            logging.FileHandler(log_file, mode='w'),  # 'w' mode overwrites the file
            # Console handler
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration of logging
    )
    
    # Get the facebeak logger specifically
    logger = logging.getLogger('facebeak')
    logger.setLevel(logging.INFO)  # Ensure level is set correctly
    
    # Log startup information
    logger.info("=== Starting new session ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    return logger 