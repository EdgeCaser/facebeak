import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='extract_training.log', force_console=False):
    """Configure logging for both GUI and command-line versions"""
    try:
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8',
            delay=True  # Don't create file until first write
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                file_handler,
                logging.StreamHandler()
            ],
            force=True  # Override any existing configuration
        )

        # Get logger for the calling module
        logger = logging.getLogger(__name__)
        
        # Log startup with module name
        caller = sys._getframe(1).f_globals.get('__name__', 'unknown')
        logger.info(f"Logging configured for {caller}. Log file: {os.path.abspath(log_file)}")
        
        return logger
    except Exception as e:
        # If anything goes wrong, at least set up console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()],
            force=True
        )
        logging.error(f"Failed to set up file logging: {e}")
        return logging.getLogger(__name__) 