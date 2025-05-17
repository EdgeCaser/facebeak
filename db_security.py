import os
import sqlite3
from cryptography.fernet import Fernet
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_encryption_key():
    """Get or create the encryption key."""
    key_path = 'crow_embeddings.key'
    try:
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return f.read()
        key = Fernet.generate_key()
        with open(key_path, 'wb') as f:
            f.write(key)
        return key
    except Exception as e:
        logger.error(f"Error with encryption key: {e}")
        raise

def secure_database_connection(db_path='crow_embeddings.db'):
    """Create a database connection with proper error handling."""
    try:
        # Create connection with proper settings
        conn = sqlite3.connect(db_path)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set busy timeout to avoid database locks
        conn.execute("PRAGMA busy_timeout = 5000")
        
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise 