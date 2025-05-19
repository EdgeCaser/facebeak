import logging
from pathlib import Path
import os
import sys
import torch

def setup_logging():
    """Set up logging configuration."""
    # Get log directory from environment or use default
    log_dir = Path(os.environ.get('LOG_DIR', 'logs'))
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