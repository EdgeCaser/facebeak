import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sqlite3
from cryptography.fernet import Fernet
import sys

# Import the database security module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_security import (
    derive_key_from_password,
    get_encryption_key,
    secure_database_connection,
    verify_database_integrity
)

class TestDatabaseSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_key_dir = os.path.join(cls.test_dir, "test_keys")
        os.makedirs(cls.test_key_dir, exist_ok=True)
        
    def setUp(self):
        """Set up each test with fresh environment."""
        self.test_db_path = os.path.join(self.test_dir, f"test_db_{id(self)}.db")
        
        # Patch the key directory environment variable
        self.key_dir_patcher = patch.dict(os.environ, {
            'FACEBEAK_KEY_DIR': self.test_key_dir
        })
        self.key_dir_patcher.start()
        
        # Clean up any existing key files
        key_path = Path(self.test_key_dir) / 'crow_embeddings.key'
        salt_path = Path(self.test_key_dir) / 'crow_embeddings.salt'
        if key_path.exists():
            key_path.unlink()
        if salt_path.exists():
            salt_path.unlink()
            
    def tearDown(self):
        """Clean up after each test."""
        self.key_dir_patcher.stop()
        
        # Clean up test database
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass
                
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_derive_key_from_password(self):
        """Test key derivation from password."""
        password = "test_password_123"
        
        # Test key derivation with new salt
        key1, salt1 = derive_key_from_password(password)
        self.assertIsInstance(key1, bytes)
        self.assertIsInstance(salt1, bytes)
        self.assertEqual(len(salt1), 16)  # Salt should be 16 bytes
        
        # Test key derivation with same salt produces same key
        key2, salt2 = derive_key_from_password(password, salt1)
        self.assertEqual(key1, key2)
        self.assertEqual(salt1, salt1)  # Salt should be unchanged
        
        # Test different passwords produce different keys
        key3, salt3 = derive_key_from_password("different_password", salt1)
        self.assertNotEqual(key1, key3)
        
        # Verify key can be used with Fernet
        fernet = Fernet(key1)
        test_data = b"test encryption data"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        self.assertEqual(test_data, decrypted)
    
    def test_get_encryption_key_new(self):
        """Test encryption key generation for new installation."""
        with patch('getpass.getpass', return_value='test_password_123'):
            with patch('sys.stdin.isatty', return_value=True):
                key = get_encryption_key()
        
        self.assertIsInstance(key, bytes)
        
        # Verify key files were created
        key_path = Path(self.test_key_dir) / 'crow_embeddings.key'
        salt_path = Path(self.test_key_dir) / 'crow_embeddings.salt'
        
        self.assertTrue(key_path.exists())
        self.assertTrue(salt_path.exists())
        
        # Verify key can be used with Fernet
        fernet = Fernet(key)
        test_data = b"test data"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        self.assertEqual(test_data, decrypted)
    
    def test_get_encryption_key_existing(self):
        """Test loading existing encryption key."""
        # First create a key
        with patch('getpass.getpass', return_value='test_password_123'):
            with patch('sys.stdin.isatty', return_value=True):
                key1 = get_encryption_key()
        
        # Get key again - should load existing
        key2 = get_encryption_key()
        
        self.assertEqual(key1, key2)
    
    def test_get_encryption_key_test_mode(self):
        """Test encryption key generation in test mode."""
        key = get_encryption_key(test_mode=True)
        
        self.assertIsInstance(key, bytes)
        
        # Verify key can be used with Fernet
        fernet = Fernet(key)
        test_data = b"test data"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        self.assertEqual(test_data, decrypted)
    
    def test_get_encryption_key_invalid_existing(self):
        """Test handling of invalid existing key."""
        # Create invalid key file
        key_path = Path(self.test_key_dir) / 'crow_embeddings.key'
        salt_path = Path(self.test_key_dir) / 'crow_embeddings.salt'
        
        with open(key_path, 'wb') as f:
            f.write(b'invalid_key_data')
        with open(salt_path, 'wb') as f:
            f.write(b'invalid_salt_data')
        
        # Should generate new key when existing is invalid
        with patch('getpass.getpass', return_value='new_password_456'):
            with patch('sys.stdin.isatty', return_value=True):
                key = get_encryption_key()
        
        self.assertIsInstance(key, bytes)
        
        # Verify new key works
        fernet = Fernet(key)
        test_data = b"test data"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        self.assertEqual(test_data, decrypted)
    
    def test_secure_database_connection_new_db(self):
        """Test secure database connection with new database."""
        # Create a new database
        conn = sqlite3.connect(self.test_db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.close()
        
        # Test secure connection
        secure_conn = secure_database_connection(self.test_db_path)
        
        self.assertIsNotNone(secure_conn)
        
        # Test that PRAGMA settings were applied
        cursor = secure_conn.cursor()
        
        # Check foreign keys are enabled
        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        self.assertEqual(result[0], 1)
        
        # Check WAL mode
        cursor.execute("PRAGMA journal_mode")
        result = cursor.fetchone()
        self.assertEqual(result[0], 'wal')
        
        secure_conn.close()
    
    def test_secure_database_connection_missing_file(self):
        """Test secure database connection with missing file."""
        missing_path = os.path.join(self.test_dir, "nonexistent.db")
        
        with self.assertRaises(FileNotFoundError):
            secure_database_connection(missing_path)
    
    @patch('os.access')
    def test_secure_database_connection_permission_error(self, mock_access):
        """Test secure database connection with permission errors."""
        # Create a database file
        conn = sqlite3.connect(self.test_db_path)
        conn.close()
        
        # Mock permission error
        mock_access.return_value = False
        
        with self.assertRaises(PermissionError):
            secure_database_connection(self.test_db_path)
    
    def test_verify_database_integrity_good_db(self):
        """Test database integrity verification with good database."""
        # Create a valid database
        conn = sqlite3.connect(self.test_db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test (name) VALUES ('test')")
        conn.commit()
        conn.close()
        
        # Should pass integrity check
        result = verify_database_integrity(self.test_db_path)
        self.assertTrue(result)
    
    def test_verify_database_integrity_corrupted_db(self):
        """Test database integrity verification with corrupted database."""
        # Create a corrupted database file
        with open(self.test_db_path, 'wb') as f:
            f.write(b'This is not a valid SQLite database')
        
        # Should fail integrity check
        with self.assertRaises(Exception):
            verify_database_integrity(self.test_db_path)
    
    def test_file_permissions(self):
        """Test that key files have correct permissions."""
        with patch('getpass.getpass', return_value='test_password_123'):
            with patch('sys.stdin.isatty', return_value=True):
                get_encryption_key()
        
        key_path = Path(self.test_key_dir) / 'crow_embeddings.key'
        salt_path = Path(self.test_key_dir) / 'crow_embeddings.salt'
        
        # Check that files exist
        self.assertTrue(key_path.exists())
        self.assertTrue(salt_path.exists())
        
        # On Unix systems, check permissions (skip on Windows)
        if os.name != 'nt':
            key_stat = key_path.stat()
            salt_stat = salt_path.stat()
            
            # Files should be readable/writable by owner only (0o600)
            self.assertEqual(oct(key_stat.st_mode)[-3:], '600')
            self.assertEqual(oct(salt_stat.st_mode)[-3:], '600')
    
    def test_password_validation(self):
        """Test password validation requirements."""
        # Test short password
        with patch('getpass.getpass', side_effect=['short', 'longer_password']):
            with patch('sys.stdin.isatty', return_value=True):
                key = get_encryption_key()
        
        # Should still succeed after retry
        self.assertIsInstance(key, bytes)
    
    def test_non_interactive_mode(self):
        """Test key generation in non-interactive mode."""
        with patch('sys.stdin.isatty', return_value=False):
            key = get_encryption_key()
        
        # Should use default password and succeed
        self.assertIsInstance(key, bytes)
        
        # Verify key works
        fernet = Fernet(key)
        test_data = b"test data"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        self.assertEqual(test_data, decrypted)
    
    def test_key_directory_creation(self):
        """Test that key directory is created if it doesn't exist."""
        # Use a non-existent directory
        new_key_dir = os.path.join(self.test_dir, "new_key_dir")
        
        with patch.dict(os.environ, {'FACEBEAK_KEY_DIR': new_key_dir}):
            with patch('getpass.getpass', return_value='test_password_123'):
                with patch('sys.stdin.isatty', return_value=True):
                    key = get_encryption_key()
        
        # Directory should be created
        self.assertTrue(os.path.exists(new_key_dir))
        
        # Key files should exist
        key_path = Path(new_key_dir) / 'crow_embeddings.key'
        salt_path = Path(new_key_dir) / 'crow_embeddings.salt'
        
        self.assertTrue(key_path.exists())
        self.assertTrue(salt_path.exists())
    
    def test_database_pragma_settings(self):
        """Test that all required PRAGMA settings are applied."""
        # Create a test database
        conn = sqlite3.connect(self.test_db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.close()
        
        # Get secure connection
        secure_conn = secure_database_connection(self.test_db_path)
        cursor = secure_conn.cursor()
        
        # Test all PRAGMA settings
        pragma_tests = [
            ("PRAGMA foreign_keys", 1),
            ("PRAGMA journal_mode", "wal"),
            ("PRAGMA synchronous", 1),  # NORMAL = 1
            ("PRAGMA page_size", 4096),
            ("PRAGMA temp_store", 2),  # MEMORY = 2
            ("PRAGMA recursive_triggers", 1),
        ]
        
        for pragma_query, expected in pragma_tests:
            cursor.execute(pragma_query)
            result = cursor.fetchone()
            self.assertEqual(result[0], expected, f"Failed PRAGMA test: {pragma_query}")
        
        secure_conn.close()

if __name__ == '__main__':
    unittest.main() 