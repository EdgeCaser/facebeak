import sqlite3
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseEncryption:
    def __init__(self, db_path='crow_embeddings.db'):
        self.db_path = db_path
        self.key = None
        self.fernet = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with a key from environment variable or create new one."""
        try:
            # Check if key file exists
            key_path = Path(self.db_path).with_suffix('.key')
            if key_path.exists():
                # Load existing key
                with open(key_path, 'rb') as f:
                    stored_salt = f.read(16)
                    stored_key = f.read()
                
                # Get password from environment variable
                password = os.getenv('DB_PASSWORD')
                if not password:
                    raise ValueError("DB_PASSWORD not found in environment variables")
                
                key = self._derive_key(password.encode(), stored_salt)
                if key != stored_key:
                    raise ValueError("Invalid database password in environment variables")
            else:
                # Generate new key
                password = os.getenv('DB_PASSWORD')
                if not password:
                    # Generate a random password if none exists
                    password = base64.urlsafe_b64encode(os.urandom(32)).decode()
                    # Save it to .env file
                    with open('.env', 'a') as f:
                        f.write(f"\nDB_PASSWORD={password}\n")
                    logger.info("Generated new database password and saved to .env file")
                
                salt = os.urandom(16)
                key = self._derive_key(password.encode(), salt)
                
                # Save salt and key
                with open(key_path, 'wb') as f:
                    f.write(salt)
                    f.write(key)
            
            self.key = key
            self.fernet = Fernet(base64.urlsafe_b64encode(key[:32]))
            logger.info("Database encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {str(e)}")
            raise
    
    def _derive_key(self, password, salt):
        """Derive encryption key from password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password)
    
    def encrypt_database(self):
        """Encrypt the entire database file."""
        try:
            # Read database file
            with open(self.db_path, 'rb') as f:
                data = f.read()
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data)
            
            # Write encrypted data
            encrypted_path = f"{self.db_path}.encrypted"
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Backup original
            backup_path = f"{self.db_path}.backup"
            if os.path.exists(self.db_path):
                os.replace(self.db_path, backup_path)
            
            # Replace with encrypted version
            os.replace(encrypted_path, self.db_path)
            
            logger.info("Database encrypted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt database: {str(e)}")
            return False
    
    def decrypt_database(self):
        """Decrypt the database file for use."""
        try:
            # Read encrypted database
            with open(self.db_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Write decrypted data to temporary file
            temp_path = f"{self.db_path}.temp"
            with open(temp_path, 'wb') as f:
                f.write(decrypted_data)
            
            # Replace encrypted file with decrypted version
            os.replace(temp_path, self.db_path)
            
            logger.info("Database decrypted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to decrypt database: {str(e)}")
            return False
    
    def backup_database(self, backup_dir='backups'):
        """Create an encrypted backup of the database."""
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"{Path(self.db_path).stem}_{timestamp}.db.encrypted")
            
            # Read and encrypt database
            with open(self.db_path, 'rb') as f:
                data = f.read()
            encrypted_data = self.fernet.encrypt(data)
            
            # Write encrypted backup
            with open(backup_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database backup: {str(e)}")
            return False

def secure_database_connection(db_path='crow_embeddings.db'):
    """Create a secure database connection with automatic encryption/decryption."""
    encryption = DatabaseEncryption(db_path)
    
    # Check if database exists but isn't encrypted yet
    if os.path.exists(db_path) and not os.path.exists(f"{db_path}.key"):
        logger.info("Found existing unencrypted database. Encrypting...")
        try:
            # First create a temporary copy
            temp_path = f"{db_path}.temp"
            import shutil
            shutil.copy2(db_path, temp_path)
            
            # Verify the copy is a valid database
            try:
                test_conn = sqlite3.connect(temp_path)
                test_conn.close()
            except sqlite3.Error:
                logger.error("Database file is corrupted. Creating new database.")
                os.remove(temp_path)
                os.remove(db_path)
                return _create_new_database(db_path, encryption)
            
            # Create backup of original
            backup_path = f"{db_path}.backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(db_path, backup_path)
            
            # Move temp to main location
            os.rename(temp_path, db_path)
            
            # Now encrypt the database
            if not encryption.encrypt_database():
                # If encryption fails, restore from backup
                if os.path.exists(backup_path):
                    os.remove(db_path)
                    os.rename(backup_path, db_path)
                raise RuntimeError("Failed to encrypt database")
            
            logger.info("Successfully encrypted existing database")
            
        except Exception as e:
            logger.error(f"Error during database encryption: {str(e)}")
            # Try to restore from backup if it exists
            if os.path.exists(backup_path):
                if os.path.exists(db_path):
                    os.remove(db_path)
                os.rename(backup_path, db_path)
            raise RuntimeError("Failed to encrypt database")
    
    # Try to decrypt database for use
    if os.path.exists(db_path):
        try:
            if not encryption.decrypt_database():
                raise RuntimeError("Failed to decrypt database")
        except Exception as e:
            logger.error(f"Error decrypting database: {str(e)}")
            # If decryption fails, try to create new database
            return _create_new_database(db_path, encryption)
    else:
        logger.info("No existing database found. Will create new encrypted database.")
        return _create_new_database(db_path, encryption)
    
    try:
        # Create connection
        conn = sqlite3.connect(db_path)
        
        # Add cleanup handler to encrypt database when done
        def cleanup():
            try:
                conn.close()
                encryption.encrypt_database()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
        
        # Register cleanup
        import atexit
        atexit.register(cleanup)
        
        return conn
        
    except Exception as e:
        logger.error(f"Failed to create secure database connection: {str(e)}")
        # Ensure database is encrypted even if connection fails
        encryption.encrypt_database()
        raise

def _create_new_database(db_path, encryption):
    """Create a new encrypted database."""
    try:
        # Create new database
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''
        CREATE TABLE IF NOT EXISTS crows (
            id INTEGER PRIMARY KEY,
            name TEXT,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            total_sightings INTEGER DEFAULT 0
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS crow_embeddings (
            id INTEGER PRIMARY KEY,
            crow_id INTEGER,
            embedding BLOB,
            video_path TEXT,
            frame_number INTEGER,
            timestamp TIMESTAMP,
            confidence REAL,
            FOREIGN KEY (crow_id) REFERENCES crows (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
        # Encrypt the new database
        if not encryption.encrypt_database():
            raise RuntimeError("Failed to encrypt new database")
        
        # Decrypt for use
        if not encryption.decrypt_database():
            raise RuntimeError("Failed to decrypt new database")
        
        # Create final connection
        conn = sqlite3.connect(db_path)
        
        # Add cleanup handler
        def cleanup():
            try:
                conn.close()
                encryption.encrypt_database()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
        
        import atexit
        atexit.register(cleanup)
        
        return conn
        
    except Exception as e:
        logger.error(f"Failed to create new database: {str(e)}")
        if os.path.exists(db_path):
            os.remove(db_path)
        raise RuntimeError("Failed to create new database") 