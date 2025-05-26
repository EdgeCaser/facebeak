import os
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging
from pathlib import Path
import secrets
import getpass
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def derive_key_from_password(password, salt=None):
    """Derive an encryption key from a password using PBKDF2."""
    if salt is None:
        salt = secrets.token_bytes(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    # Derive raw 32-byte key and encode it for Fernet
    raw_key = kdf.derive(password.encode())
    fernet_key = base64.urlsafe_b64encode(raw_key)
    return fernet_key, salt

def get_encryption_key(test_mode=False):
    """Get or create the encryption key."""
    key_dir_env = os.environ.get("FACEBEAK_KEY_DIR")
    if key_dir_env:
         key_dir = Path(key_dir_env)
    else:
         key_dir = Path.home() / '.facebeak' / 'keys'
    key_dir.mkdir(parents=True, exist_ok=True)
    
    key_path = key_dir / 'crow_embeddings.key'
    salt_path = key_dir / 'crow_embeddings.salt'
    
    try:
        if key_path.exists() and salt_path.exists():
            # Read existing key and salt
            with open(key_path, 'rb') as f:
                key = f.read()
            with open(salt_path, 'rb') as f:
                salt = f.read()
                
            # Verify key is valid for Fernet (should be base64-encoded)
            try:
                Fernet(key)
                return key
            except Exception:
                logger.warning("Existing key is invalid, generating new key")
                key_path.unlink()
                salt_path.unlink()
        
        # Generate new key
        logger.info("Generating new encryption key")
        
        if test_mode:
            # Use a test password in test mode
            password = "test_password_123"
        else:
            # Get password from user
            if sys.stdin.isatty():
                password = getpass.getpass("Enter password for database encryption (min 8 chars): ")
                while len(password) < 8:
                    password = getpass.getpass("Password too short. Enter password (min 8 chars): ")
            else:
                # If stdin is not a terminal (e.g., in tests), use a default password
                password = "default_password_123"
            
        key, salt = derive_key_from_password(password)
        
        # Save key and salt
        with open(key_path, 'wb') as f:
            f.write(key)
        with open(salt_path, 'wb') as f:
            f.write(salt)
            
        # Set restrictive permissions
        key_path.chmod(0o600)
        salt_path.chmod(0o600)
        
        return key
    except Exception as e:
        logger.error(f"Error with encryption key: {e}")
        raise

def secure_database_connection(db_path):
    """Create a secure database connection with proper settings."""
    try:
        # Ensure database directory exists
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if database file exists
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
            
        # Check if database file is readable
        if not os.access(db_path, os.R_OK):
            raise PermissionError(f"Database file not readable: {db_path}")
            
        # Check if database directory is writable
        if not os.access(db_path.parent, os.W_OK):
            raise PermissionError(f"Database directory not writable: {db_path.parent}")
        
        # Create connection with proper settings
        conn = sqlite3.connect(str(db_path), timeout=30.0)  # Increased timeout
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set busy timeout to avoid database locks
        conn.execute("PRAGMA busy_timeout = 5000")
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Set synchronous mode for better durability
        conn.execute("PRAGMA synchronous = NORMAL")
        
        # Set page size for better performance
        conn.execute("PRAGMA page_size = 4096")
        
        # Set cache size
        conn.execute("PRAGMA cache_size = -2000")  # Use 2MB of cache
        
        # Set temp store to memory
        conn.execute("PRAGMA temp_store = MEMORY")
        
        # Set mmap size for better performance
        conn.execute("PRAGMA mmap_size = 30000000000")  # 30GB
        
        # Enable recursive triggers
        conn.execute("PRAGMA recursive_triggers = ON")
        
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def verify_database_integrity(db_path):
    """Verify database integrity."""
    conn = None
    try:
        conn = secure_database_connection(db_path)
        cursor = conn.cursor()
        
        # Check database integrity
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        if result[0] != "ok":
            raise ValueError(f"Database integrity check failed: {result[0]}")
            
        # Check foreign key constraints
        cursor.execute("PRAGMA foreign_key_check")
        if cursor.fetchone():
            raise ValueError("Foreign key constraint violation detected")
            
        # Check for corruption
        cursor.execute("PRAGMA quick_check")
        result = cursor.fetchone()
        if result[0] != "ok":
            raise ValueError(f"Database corruption detected: {result[0]}")
            
        logger.info("Database integrity check passed")
        return True
    except Exception as e:
        logger.error(f"Database integrity check failed: {e}")
        raise
    finally:
        if conn:
            conn.close() 