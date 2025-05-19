import logging
from pathlib import Path
import os
import sys
import torch

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Use a single log file that gets overwritten
    log_file = log_dir / 'training.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            # File handler - overwrites the file on each run
            logging.FileHandler(log_file, mode='w'),  # 'w' mode overwrites the file
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=== Starting new training session ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    return logger 