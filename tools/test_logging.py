import logging
from pathlib import Path

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'test_logging.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info('TEST: Logging is working!')

print("Check logs/test_logging.log for the test message.") 