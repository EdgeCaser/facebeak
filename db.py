import sqlite3
import numpy as np
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from db_security import secure_database_connection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = 'crow_embeddings.db'

def initialize_database():
    """Initialize the database."""
    try:
        # Create connection
        conn = secure_database_connection(DB_PATH)
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS crows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_sightings INTEGER DEFAULT 0
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS crow_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crow_id INTEGER,
            embedding BLOB NOT NULL,
            video_path TEXT,
            frame_number INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence FLOAT,
            FOREIGN KEY (crow_id) REFERENCES crows(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_connection():
    """Get a database connection."""
    return secure_database_connection(DB_PATH)

def save_crow_embedding(embedding, video_path=None, frame_number=None, confidence=1.0):
    """Save a crow embedding and try to match it with existing crows."""
    conn = get_connection()
    c = conn.cursor()
    
    try:
        # Try to find matching crow
        crow_id = find_matching_crow(
            embedding,
            threshold=0.6,
            video_path=video_path,
            frame_number=frame_number
        )
        
        if crow_id is None:
            # Create new crow entry
            c.execute('INSERT INTO crows (first_seen) VALUES (CURRENT_TIMESTAMP)')
            crow_id = c.lastrowid
        
        # Update crow's last seen time and increment sightings
        c.execute('''
            UPDATE crows 
            SET last_seen = CURRENT_TIMESTAMP,
                total_sightings = total_sightings + 1
            WHERE id = ?
        ''', (crow_id,))
        
        # Save the embedding
        embedding_blob = embedding.astype(np.float32).tobytes()
        c.execute('''
            INSERT INTO crow_embeddings 
            (crow_id, embedding, video_path, frame_number, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (crow_id, embedding_blob, video_path, frame_number, confidence))
        
        conn.commit()
        return crow_id
        
    finally:
        conn.close()

def find_matching_crow(embedding, threshold=0.6, video_path=None, frame_number=None):  # Lowered threshold
    """
    Find a matching crow in the database based on embedding similarity and temporal consistency.
    Returns crow_id if match found, None otherwise.
    """
    conn = get_connection()
    c = conn.cursor()
    
    # Get recent embeddings from the same video first
    if video_path:
        c.execute('''
            WITH recent_embeddings AS (
                SELECT crow_id, embedding, frame_number,
                       ROW_NUMBER() OVER (PARTITION BY crow_id ORDER BY frame_number DESC) as rn
                FROM crow_embeddings
                WHERE video_path = ? AND frame_number < ?
                ORDER BY frame_number DESC
                LIMIT 100
            )
            SELECT crow_id, embedding, frame_number
            FROM recent_embeddings
            WHERE rn = 1
        ''', (video_path, frame_number))
        recent_rows = c.fetchall()
        
        # Check recent embeddings first with a stricter threshold
        for crow_id, emb_blob, frame_num in recent_rows:
            known_emb = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = 1 - cosine(embedding, known_emb)
            # Use stricter threshold for same-video matches
            if similarity > 0.7:  # Higher threshold for same video
                conn.close()
                return crow_id
    
    # If no match in same video, check all known crows
    c.execute('''
        WITH latest_embeddings AS (
            SELECT crow_id, embedding, timestamp,
                   ROW_NUMBER() OVER (PARTITION BY crow_id ORDER BY timestamp DESC) as rn
            FROM crow_embeddings
        )
        SELECT crow_id, embedding, timestamp
        FROM latest_embeddings 
        WHERE rn = 1
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return None
    
    # Compare with all known crows
    best_match = None
    best_score = 0
    
    for crow_id, emb_blob, timestamp in rows:
        known_emb = np.frombuffer(emb_blob, dtype=np.float32)
        # Use cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(embedding, known_emb)
        
        # Adjust threshold based on crow's history
        crow_history = get_crow_history(crow_id)
        if crow_history:
            # More lenient threshold for crows with more sightings
            adjusted_threshold = threshold - (min(crow_history['total_sightings'], 50) * 0.002)
            adjusted_threshold = max(0.5, adjusted_threshold)  # Don't go below 0.5
        else:
            adjusted_threshold = threshold
        
        if similarity > adjusted_threshold and similarity > best_score:
            best_score = similarity
            best_match = crow_id
    
    return best_match

def get_crow_history(crow_id):
    """Get the sighting history for a specific crow."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT c.id, c.first_seen, c.last_seen, c.total_sightings,
               GROUP_CONCAT(DISTINCT ce.video_path) as videos,
               COUNT(DISTINCT ce.video_path) as video_count
        FROM crows c
        LEFT JOIN crow_embeddings ce ON c.id = ce.crow_id
        WHERE c.id = ?
        GROUP BY c.id
    ''', (crow_id,))
    
    row = c.fetchone()
    conn.close()
    
    if row:
        crow_id, first_seen, last_seen, total_sightings, videos, video_count = row
        return {
            'id': crow_id,
            'first_seen': first_seen,
            'last_seen': last_seen,
            'total_sightings': total_sightings,
            'videos': videos.split(',') if videos else [],
            'video_count': video_count
        }
    return None

def get_all_crows():
    """Get summary of all known crows."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT c.id, c.name, c.first_seen, c.last_seen, c.total_sightings,
               COUNT(DISTINCT ce.video_path) as video_count
        FROM crows c
        LEFT JOIN crow_embeddings ce ON c.id = ce.crow_id
        GROUP BY c.id
        ORDER BY c.last_seen DESC
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    return [{
        'id': row[0],
        'name': row[1] if row[1] else f"Crow {row[0]}",  # Provide default name if none exists
        'first_seen': row[2],
        'last_seen': row[3],
        'total_sightings': row[4],
        'video_count': row[5]
    } for row in rows]

def update_crow_name(crow_id, name):
    """Update the name of a crow."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('UPDATE crows SET name = ? WHERE id = ?', (name, crow_id))
    conn.commit()
    conn.close()

def get_crow_embeddings(crow_id):
    """Get all embeddings for a specific crow."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT embedding, video_path, frame_number, timestamp, confidence
        FROM crow_embeddings
        WHERE crow_id = ?
        ORDER BY timestamp DESC
    ''', (crow_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [{
        'embedding': np.frombuffer(row[0], dtype=np.float32),
        'video_path': row[1],
        'frame_number': row[2],
        'timestamp': row[3],
        'confidence': row[4]
    } for row in rows]

def backup_database():
    """Create a backup of the database."""
    try:
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f'crow_embeddings_{timestamp}.db')
        
        # Copy the database file
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        
        logger.info(f"Database backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None

def clear_database():
    """Clear all data from the database."""
    conn = None
    try:
        # Get a connection
        conn = get_connection()
        if not conn:
            logger.error("Failed to get database connection")
            return False
            
        # Disable foreign key constraints
        conn.execute("PRAGMA foreign_keys = OFF")
        
        # Delete all entries from tables
        conn.execute("DELETE FROM crow_embeddings")
        conn.execute("DELETE FROM crows")
        
        # Reset autoincrement counters
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('crows', 'crow_embeddings')")
        
        # Re-enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Commit changes
        conn.commit()
        logger.info("Database cleared successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

# Initialize database on module import
initialize_database() 