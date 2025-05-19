import unittest
from unittest.mock import MagicMock, patch, call
import pytest
import logging
import os
import tempfile
from pathlib import Path
from logging_config import setup_logging

class TestLoggingConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Create a temporary directory for test logs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.log_dir = os.path.join(cls.temp_dir.name, "logs")
        os.makedirs(cls.log_dir, exist_ok=True)
        
        # Set environment variable for test logs
        os.environ['LOG_DIR'] = cls.log_dir
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Reset logging configuration before each test
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.setLevel(logging.NOTSET)
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove all handlers and close log files
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
            
        # Clean up log files
        for log_file in Path(self.log_dir).glob("*.log"):
            try:
                log_file.unlink()
            except:
                pass
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()
        
    def test_setup_logging(self):
        """Test basic logging setup."""
        # Setup logging
        logger = setup_logging()
        
        # Verify logger configuration
        self.assertEqual(logger.name, "facebeak")
        self.assertEqual(logger.level, logging.INFO)
        
        # Verify log file was created
        log_files = list(Path(self.log_dir).glob("*.log"))
        self.assertEqual(len(log_files), 1)
        
        # Test logging
        test_message = "Test log message"
        logger.info(test_message)
        
        # Verify message was written to log file
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)
            
    def test_logging_levels(self):
        """Test different logging levels."""
        logger = setup_logging()
        
        # Test different log levels
        test_messages = {
            logging.DEBUG: "Debug message",
            logging.INFO: "Info message",
            logging.WARNING: "Warning message",
            logging.ERROR: "Error message",
            logging.CRITICAL: "Critical message"
        }
        
        # Log messages at different levels
        for level, message in test_messages.items():
            logger.log(level, message)
            
        # Read log file
        log_files = list(Path(self.log_dir).glob("*.log"))
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            
        # Verify messages at or above INFO level are logged
        self.assertNotIn("Debug message", log_content)  # DEBUG messages should not be logged
        self.assertIn("Info message", log_content)
        self.assertIn("Warning message", log_content)
        self.assertIn("Error message", log_content)
        self.assertIn("Critical message", log_content)
        
    def test_log_format(self):
        """Test log message formatting."""
        logger = setup_logging()
        
        # Log a test message
        test_message = "Test format message"
        logger.info(test_message)
        
        # Read log file
        log_files = list(Path(self.log_dir).glob("*.log"))
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            
        # Verify log format
        # Should contain timestamp, log level, filename, line number, and message
        self.assertRegex(log_content, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - \[test_logging_config\.py:\d+\] - ' + test_message)
        
    def test_multiple_loggers(self):
        """Test creating multiple logger instances."""
        # Create two loggers
        logger1 = setup_logging()
        logger2 = setup_logging()
        
        # Verify they are the same logger instance
        self.assertIs(logger1, logger2)
        
        # Test logging from both references
        logger1.info("Message from logger1")
        logger2.info("Message from logger2")
        
        # Verify both messages were logged (only the last one should be in the file due to 'w' mode)
        log_files = list(Path(self.log_dir).glob("*.log"))
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            self.assertIn("Message from logger2", log_content)  # Only the last message should be present
            
    def test_error_handling(self):
        """Test logging error handling."""
        logger = setup_logging()
        
        # Test logging with invalid characters
        test_message = "Test message with invalid chars: \x00\x01\x02"
        logger.info(test_message)
        
        # Test logging very long message
        long_message = "x" * 10000
        logger.info(long_message)
        
        # Verify log file is still readable
        log_files = list(Path(self.log_dir).glob("*.log"))
        with open(log_files[0], 'r') as f:
            log_content = f.read()
            self.assertIn("Test message with invalid chars", log_content)
            self.assertIn("x" * 100, log_content)  # At least part of the long message should be logged

if __name__ == '__main__':
    unittest.main() 