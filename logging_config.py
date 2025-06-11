import logging
from pathlib import Path
import os
import sys
import json

# Load configuration at the start of the script
CONFIG = {}
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logging.getLogger(__name__).info("config.json not found in logging_config.py. Using default log path logic.")
except json.JSONDecodeError:
    logging.getLogger(__name__).info("Error decoding config.json in logging_config.py. Using default log path logic.")

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
    
    # Log PyTorch info if available (import only when needed)
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.info("PyTorch not available")
    except Exception as e:
        logger.warning(f"Error getting PyTorch info: {e}")
    
    return logger 

def log_system_info(logger):
    """Log comprehensive system information for ML applications."""
    logger.info("=== System Information ===")
    
    # Log available ML libraries
    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        logger.warning("OpenCV not available")
    
    try:
        import kivy
        logger.info(f"Kivy version: {kivy.__version__}")
    except ImportError:
        logger.warning("Kivy not available")
    
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError:
        logger.warning("NumPy not available")
    
    try:
        import ultralytics
        logger.info(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        logger.warning("Ultralytics not available")
    
    logger.info("=== End System Information ===")

def log_torch_info_detailed(logger):
    """Log detailed PyTorch information."""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    except ImportError:
        logger.warning("PyTorch not available")
    except Exception as e:
        logger.warning(f"Error getting detailed PyTorch info: {e}") 