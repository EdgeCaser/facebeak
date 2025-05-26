import sqlite3
import numpy as np
import os
from datetime import datetime
from scipy.spatial.distance import cosine
from db_security import secure_database_connection
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_path():
    """Get the database path from environment variable or use default."""
    db_path = os.environ.get('CROW_DB_PATH')
    if db_path:
        return Path(db_path)
    # Use default path in user's home directory
    return Path.home() / '.facebeak' / 'crow_embeddings.db'

# Ensure database directory exists
def ensure_db_dir():
    """Ensure the database directory exists."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path

# Get database path
DB_PATH = ensure_db_dir()

def initialize_database():
    """Initialize the database."""
    try:
        # Create connection
        conn = secure_database_connection(str(DB_PATH))
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
            segment_id INTEGER,
            FOREIGN KEY (crow_id) REFERENCES crows(id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS behavioral_markers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            frame_number INTEGER,
            marker_type TEXT NOT NULL,
            confidence FLOAT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (segment_id) REFERENCES crow_embeddings(id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS image_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            confidence FLOAT,
            reviewer_notes TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_training_data BOOLEAN DEFAULT 1
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
    return secure_database_connection(str(DB_PATH))

def save_crow_embedding(embedding, video_path=None, frame_number=None, confidence=1.0):
    """Save a crow embedding and try to match it with existing crows."""
    conn = get_connection()
    c = conn.cursor()
    
    try:
        # Start transaction
        conn.execute("BEGIN TRANSACTION")
        
        # Convert embedding to numpy array if it's a tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        elif not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Ensure embedding is float32
        embedding = embedding.astype(np.float32)
        
        # Try to find matching crow
        crow_id = find_matching_crow(
            embedding,
            threshold=0.6,
            video_path=video_path,
            frame_number=frame_number
        )
        
        # Verify if crow exists if we got an ID
        if crow_id is not None:
            c.execute('SELECT id FROM crows WHERE id = ?', (crow_id,))
            if not c.fetchone():
                logger.warning(f"Crow ID {crow_id} returned by find_matching_crow but not found in database. Creating new crow.")
                crow_id = None
        
        if crow_id is None:
            # Create new crow entry
            c.execute('INSERT INTO crows (first_seen, last_seen, total_sightings) VALUES (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)')
            crow_id = c.lastrowid
            logger.info(f"Created new crow with ID {crow_id}")
        else:
            # Update crow's last seen time and increment sightings
            c.execute('''
                UPDATE crows 
                SET last_seen = CURRENT_TIMESTAMP,
                    total_sightings = total_sightings + 1
                WHERE id = ?
            ''', (crow_id,))
            logger.info(f"Updated existing crow {crow_id}")
        
        # Double-check crow exists before inserting embedding
        c.execute('SELECT id FROM crows WHERE id = ?', (crow_id,))
        if not c.fetchone():
            raise ValueError(f"Crow ID {crow_id} does not exist in database after creation/update")
        
        # Save the embedding
        embedding_blob = embedding.tobytes()
        c.execute('''
            INSERT INTO crow_embeddings 
            (crow_id, embedding, video_path, frame_number, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (crow_id, embedding_blob, video_path, frame_number, confidence))
        
        # Commit transaction
        conn.commit()
        logger.info(f"Successfully saved embedding for crow {crow_id}")
        return crow_id
        
    except Exception as e:
        # Rollback transaction on error
        conn.rollback()
        logger.error(f"Error saving crow embedding: {str(e)}")
        raise
    finally:
        conn.close()

def find_matching_crow(embedding, threshold=0.6, video_path=None, frame_number=None):
    """
    Find a matching crow in the database based on embedding similarity and temporal consistency.
    Returns crow_id if match found, None otherwise.
    
    Args:
        embedding: Crow embedding vector
        threshold: Base similarity threshold (0-1)
        video_path: Path to video file (for temporal consistency)
        frame_number: Frame number in video (for temporal consistency)
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Ensure input embedding is 1-D and normalized
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Get recent embeddings from the same video first
        if video_path and frame_number is not None:
            # Look for crows seen in recent frames (within last 30 frames)
            cursor.execute('''
                WITH recent_embeddings AS (
                    SELECT ce.crow_id, ce.embedding, ce.frame_number, ce.confidence,
                           ROW_NUMBER() OVER (PARTITION BY ce.crow_id ORDER BY ce.frame_number DESC) as rn
                    FROM crow_embeddings ce
                    JOIN crows c ON ce.crow_id = c.id
                    WHERE ce.video_path = ? 
                    AND ce.frame_number < ?
                    AND ce.frame_number > ? - 30  -- Only look at recent frames
                    ORDER BY ce.frame_number DESC
                )
                SELECT crow_id, embedding, frame_number, confidence
                FROM recent_embeddings
                WHERE rn = 1
            ''', (video_path, frame_number, frame_number))
            recent_rows = cursor.fetchall()
            
            # Check recent embeddings first with a stricter threshold
            for crow_id, emb_blob, frame_num, conf in recent_rows:
                # Ensure confidence is a float
                conf = float(conf) if conf is not None else 0.0
                
                known_emb = np.frombuffer(emb_blob, dtype=np.float32).reshape(-1)
                known_emb = known_emb / np.linalg.norm(known_emb)
                
                # Skip if embedding dimensions don't match
                if len(embedding) != len(known_emb):
                    logger.debug(f"Skipping temporal match for crow {crow_id} due to dimension mismatch: {len(embedding)} vs {len(known_emb)}")
                    continue
                
                # Calculate similarity
                similarity = float(1 - cosine(embedding, known_emb))
                
                # Adjust threshold based on frame distance and confidence
                frame_distance = float(frame_number - frame_num)
                temporal_factor = float(max(0.9, 1.0 - (frame_distance / 30.0)))  # Less decay over 30 frames
                confidence_factor = float(0.9 + (conf * 0.1))  # Less weight by confidence
                
                # Use higher base threshold for temporal matches
                adjusted_threshold = float(max(0.7, threshold)) * temporal_factor * confidence_factor
                
                if similarity > adjusted_threshold:
                    logger.info(f"Found temporal match for crow {crow_id} (similarity: {similarity:.3f}, "
                              f"frame distance: {frame_distance}, adjusted threshold: {adjusted_threshold:.3f})")
                    return crow_id
        
        # If no temporal match, check all known crows
        cursor.execute('''
            WITH latest_embeddings AS (
                SELECT ce.crow_id, ce.embedding, ce.timestamp, ce.confidence,
                       ROW_NUMBER() OVER (PARTITION BY ce.crow_id ORDER BY ce.timestamp DESC) as rn
                FROM crow_embeddings ce
                JOIN crows c ON ce.crow_id = c.id
            )
            SELECT crow_id, embedding, timestamp, confidence
            FROM latest_embeddings 
            WHERE rn = 1
        ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        # Compare with all known crows
        best_match = None
        best_score = 0.0
        
        for crow_id, emb_blob, timestamp, conf in rows:
            # Ensure confidence is a float
            conf = float(conf) if conf is not None else 0.0
            
            known_emb = np.frombuffer(emb_blob, dtype=np.float32).reshape(-1)
            known_emb = known_emb / np.linalg.norm(known_emb)
            
            # Skip if embedding dimensions don't match
            if len(embedding) != len(known_emb):
                logger.debug(f"Skipping crow {crow_id} due to dimension mismatch: {len(embedding)} vs {len(known_emb)}")
                continue
            
            # Calculate similarity
            similarity = float(1 - cosine(embedding, known_emb))
            
            # Get crow history for threshold adjustment
            cursor.execute('''
                SELECT total_sightings, 
                       (julianday('now') - julianday(last_seen)) as days_since_last_seen
                FROM crows 
                WHERE id = ?
            ''', (crow_id,))
            total_sightings, days_since_last_seen = cursor.fetchone()
            
            # Convert to float to ensure scalar values
            total_sightings = float(total_sightings)
            days_since_last_seen = float(days_since_last_seen)
            
            # Adjust threshold based on crow's history
            # More lenient for crows with more sightings, but less so
            history_factor = float(1.0 - (min(total_sightings, 50.0) * 0.001))  # Up to 0.05 reduction
            
            # More lenient for recently seen crows, but less so
            recency_factor = float(1.0 - (min(days_since_last_seen, 30.0) / 30.0 * 0.05))  # Up to 0.05 reduction
            
            # Weight by confidence, but less so
            confidence_factor = float(0.95 + (conf * 0.05))  # Less weight by confidence
            
            # Use much higher base threshold for non-temporal matches
            adjusted_threshold = float(max(0.8, threshold)) * history_factor * recency_factor * confidence_factor
            adjusted_threshold = float(max(0.8, adjusted_threshold))  # Don't go below 0.8
            
            if similarity > adjusted_threshold and similarity > best_score:
                best_score = similarity
                best_match = crow_id
                logger.info(f"Found potential match for crow {crow_id} (similarity: {similarity:.3f}, "
                          f"adjusted threshold: {adjusted_threshold:.3f}, "
                          f"sightings: {total_sightings}, days since last seen: {days_since_last_seen:.1f})")
        
        return best_match
    except Exception as e:
        logger.error(f"Error finding matching crow: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_crow_history(crow_id):
    """Get the sighting history for a specific crow."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get crow info
        cursor.execute('''
            SELECT c.id, c.first_seen, c.last_seen, c.total_sightings,
                   GROUP_CONCAT(DISTINCT ce.video_path) as videos,
                   COUNT(DISTINCT ce.video_path) as video_count,
                   COUNT(ce.id) as embedding_count
            FROM crows c
            LEFT JOIN crow_embeddings ce ON c.id = ce.crow_id
            WHERE c.id = ?
            GROUP BY c.id
        ''', (crow_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        crow_id, first_seen, last_seen, total_sightings, videos, video_count, embedding_count = row
        
        # Get embeddings
        cursor.execute('''
            SELECT id, video_path, frame_number, timestamp, confidence
            FROM crow_embeddings
            WHERE crow_id = ?
            ORDER BY timestamp DESC
        ''', (crow_id,))
        
        embeddings = [{
            'id': row[0],
            'video_path': row[1],
            'frame_number': row[2],
            'timestamp': row[3],
            'confidence': row[4]
        } for row in cursor.fetchall()]
        
        return {
            'id': crow_id,
            'first_seen': first_seen,
            'last_seen': last_seen,
            'total_sightings': total_sightings,
            'videos': videos.split(',') if videos else [],
            'video_count': video_count,
            'embedding_count': embedding_count,
            'embeddings': embeddings
        }
    except Exception as e:
        logger.error(f"Error getting crow history: {e}")
        raise
    finally:
        if conn:
            conn.close()

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
    conn = None
    try:
        conn = get_connection()
        conn.execute("BEGIN TRANSACTION")
        
        # Verify crow exists
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM crows WHERE id = ?', (crow_id,))
        if not cursor.fetchone():
            raise ValueError(f"Crow ID {crow_id} does not exist")
            
        cursor.execute('UPDATE crows SET name = ? WHERE id = ?', (name, crow_id))
        conn.commit()
        logger.info(f"Updated name for crow {crow_id} to {name}")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error updating crow name: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_crow_embeddings(crow_id):
    """Get all embeddings for a specific crow."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Verify crow exists
        cursor.execute('SELECT id FROM crows WHERE id = ?', (crow_id,))
        if not cursor.fetchone():
            raise ValueError(f"Crow ID {crow_id} does not exist")
        
        cursor.execute('''
            SELECT e.embedding, e.video_path, e.frame_number, e.timestamp, e.confidence,
                   e.segment_id, e.id as embedding_id
            FROM crow_embeddings e
            WHERE e.crow_id = ?
            ORDER BY e.timestamp DESC
        ''', (crow_id,))
        
        rows = cursor.fetchall()
        return [{
            'embedding': np.frombuffer(row[0], dtype=np.float32),
            'video_path': row[1],
            'frame_number': row[2],
            'timestamp': row[3],
            'confidence': row[4],
            'segment_id': row[5],
            'embedding_id': row[6],
            'markers': get_segment_markers(row[5]) if row[5] else []  # Include behavioral markers
        } for row in rows]
    except Exception as e:
        logger.error(f"Error getting crow embeddings: {e}")
        raise
    finally:
        if conn:
            conn.close()

def add_behavioral_marker(segment_id, marker_type, details, confidence=1.0, frame_number=None):
    """Add a behavioral marker for a crow segment."""
    conn = None
    try:
        conn = get_connection()
        conn.execute("BEGIN TRANSACTION")
        cursor = conn.cursor()
        
        # Verify segment exists
        cursor.execute('SELECT id FROM crow_embeddings WHERE id = ?', (segment_id,))
        if not cursor.fetchone():
            raise ValueError(f"Segment ID {segment_id} does not exist")
            
        # Validate marker type
        if not isinstance(marker_type, str) or not marker_type.strip():
            raise ValueError("Marker type must be a non-empty string")
            
        # Validate confidence
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
            
        cursor.execute('''
            INSERT INTO behavioral_markers 
            (segment_id, frame_number, marker_type, confidence, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (segment_id, frame_number, marker_type, confidence, details))
        
        marker_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Added behavioral marker {marker_id} for segment {segment_id}")
        return marker_id
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error adding behavioral marker: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_segment_markers(segment_id):
    """Get all behavioral markers for a segment."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Verify segment exists
        cursor.execute('SELECT id FROM crow_embeddings WHERE id = ?', (segment_id,))
        if not cursor.fetchone():
            raise ValueError(f"Segment ID {segment_id} does not exist")
        
        cursor.execute('''
            SELECT marker_type, details, confidence, frame_number, timestamp
            FROM behavioral_markers 
            WHERE segment_id = ?
            ORDER BY frame_number ASC
        ''', (segment_id,))
        
        rows = cursor.fetchall()
        return [{
            'marker_type': row[0],  # Changed from 'type' to 'marker_type'
            'value': row[1],
            'confidence': row[2],
            'frame_number': row[3],
            'timestamp': row[4]
        } for row in rows]
    except Exception as e:
        logger.error(f"Error getting segment markers: {e}")
        raise
    finally:
        if conn:
            conn.close()

def backup_database():
    """Create a backup of the database."""
    try:
        backup_dir = DB_PATH.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f'crow_embeddings_{timestamp}.db'
        
        # Copy the database file
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        
        logger.info(f"Database backup created: {backup_path}")
        return str(backup_path)
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

def add_image_label(image_path, label, confidence=None, reviewer_notes=None, is_training_data=None):
    """Add or update an image label."""
    conn = None
    try:
        conn = get_connection()
        conn.execute("BEGIN TRANSACTION")
        cursor = conn.cursor()
        
        # Validate inputs
        if not os.path.exists(image_path):
            logger.warning(f"Image path does not exist: {image_path}")
            return False
            
        if not label or label not in ['crow', 'not_a_crow', 'not_sure', 'multi_crow']:
            logger.warning(f"Invalid label: {label}. Must be 'crow', 'not_a_crow', 'not_sure', or 'multi_crow'")
            return False
            
        if confidence is not None and not 0 <= confidence <= 1:
            logger.warning("Confidence must be between 0 and 1")
            return False
            
        # Implement "innocent until proven guilty" philosophy
        # If is_training_data is not explicitly set, determine based on label
        if is_training_data is None:
            # Exclude not_a_crow, multi_crow, and not_sure from training data by default
            is_training_data = (label not in ['not_a_crow', 'multi_crow', 'not_sure'])
            
        # Insert or update label
        cursor.execute('''
            INSERT OR REPLACE INTO image_labels 
            (image_path, label, confidence, reviewer_notes, is_training_data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (image_path, label, confidence, reviewer_notes, is_training_data))
        
        label_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Added/updated label for {image_path}: {label}")
        return label_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error adding image label: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_unlabeled_images(limit=20, from_directory=None):
    """
    Get a list of unlabeled crow images from the crops directory.
    
    Args:
        limit (int): Maximum number of images to return
        from_directory (str, optional): Specific directory to scan instead of default crow_crops
        
    Returns:
        list: List of image file paths that don't have labels in the database
    """
    try:
        # Use specified directory or default to crow_crops
        search_dir = from_directory if from_directory else "crow_crops"
        
        if not os.path.exists(search_dir):
            logger.warning(f"Directory {search_dir} does not exist")
            return []
        
        # Get all image files from the directory
        image_files = []
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Normalize path separators for consistent matching
                    full_path = os.path.join(root, file).replace('\\', '/')
                    image_files.append(full_path)
        
        # Get already labeled images from database (only for images in our directory)
        conn = get_connection()
        cursor = conn.cursor()
        
        if image_files:
            placeholders = ','.join('?' * len(image_files))
            cursor.execute(f"SELECT image_path FROM image_labels WHERE image_path IN ({placeholders})", image_files)
            labeled_paths = {row[0] for row in cursor.fetchall()}
        else:
            labeled_paths = set()
        
        # Filter out labeled images
        unlabeled = [path for path in image_files if path not in labeled_paths]
        
        # Shuffle and limit results
        import random
        random.shuffle(unlabeled)
        return unlabeled[:limit]
        
    except Exception as e:
        logger.error(f"Error getting unlabeled images: {e}")
        return []

def get_image_label(image_path):
    """Get the label for a specific image."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT label, confidence, reviewer_notes, timestamp, is_training_data, created_at, updated_at
            FROM image_labels 
            WHERE image_path = ?
        ''', (image_path,))
        
        row = cursor.fetchone()
        if row:
            return {
                'label': row[0],
                'confidence': row[1],
                'reviewer_notes': row[2],
                'timestamp': row[3],
                'is_training_data': bool(row[4]),
                'created_at': row[5],
                'updated_at': row[6]
            }
        return None
        
    except Exception as e:
        logger.error(f"Error getting image label: {e}")
        return None
    finally:
        if conn:
            conn.close()

def remove_from_training_data(image_path):
    """Mark an image as not suitable for training data."""
    return add_image_label(image_path, 'not_a_crow', is_training_data=False)

def get_training_data_stats(from_directory=None):
    """
    Get statistics about manually labeled training data.
    
    Args:
        from_directory (str, optional): Specific directory to scan instead of default crow_crops
        
    Returns:
        dict: Statistics about labeled images
    """
    try:
        # Use specified directory or default to crow_crops
        search_dir = from_directory if from_directory else "crow_crops"
        
        # Get all image files from the directory
        all_images = set()
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.add(os.path.join(root, file))
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all labels for images in our search directory
        if all_images:
            placeholders = ','.join('?' * len(all_images))
            cursor.execute(f"""
                SELECT label, confidence, is_training_data 
                FROM image_labels 
                WHERE image_path IN ({placeholders})
            """, list(all_images))
        else:
            # If no directory or images, return empty stats
            return {
                'crow': {'count': 0, 'avg_confidence': 0.0},
                'not_a_crow': {'count': 0, 'avg_confidence': 0.0},
                'not_sure': {'count': 0, 'avg_confidence': 0.0},
                'multi_crow': {'count': 0, 'avg_confidence': 0.0},
                'total_labeled': 0,
                'total_excluded': 0
            }
        
        # Process results
        stats = {
            'crow': {'count': 0, 'total_confidence': 0.0},
            'not_a_crow': {'count': 0, 'total_confidence': 0.0},
            'not_sure': {'count': 0, 'total_confidence': 0.0},
            'multi_crow': {'count': 0, 'total_confidence': 0.0},
            'total_labeled': 0,
            'total_excluded': 0
        }
        
        for label, confidence, is_training_data in cursor.fetchall():
            if label in stats:
                stats[label]['count'] += 1
                stats[label]['total_confidence'] += confidence or 0.0
                stats['total_labeled'] += 1
                
                if not is_training_data:
                    stats['total_excluded'] += 1
        
        # Calculate averages
        for label in ['crow', 'not_a_crow', 'not_sure', 'multi_crow']:
            count = stats[label]['count']
            if count > 0:
                stats[label]['avg_confidence'] = stats[label]['total_confidence'] / count
            else:
                stats[label]['avg_confidence'] = 0.0
            # Remove total_confidence as it's not needed in final output
            del stats[label]['total_confidence']
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting training data stats: {e}")
        return {
            'crow': {'count': 0, 'avg_confidence': 0.0},
            'not_a_crow': {'count': 0, 'avg_confidence': 0.0},
            'not_sure': {'count': 0, 'avg_confidence': 0.0},
            'multi_crow': {'count': 0, 'avg_confidence': 0.0},
            'total_labeled': 0,
            'total_excluded': 0
        }

def get_training_suitable_images(from_directory=None):
    """
    Get images suitable for training (not explicitly marked as not_a_crow).
    Implements "innocent until proven guilty" philosophy.
    
    Args:
        from_directory (str, optional): Specific directory to scan instead of default crow_crops
        
    Returns:
        list: List of image paths suitable for training
    """
    try:
        # Use specified directory or default to crow_crops
        search_dir = from_directory if from_directory else "crow_crops"
        
        if not os.path.exists(search_dir):
            logger.warning(f"Directory {search_dir} does not exist")
            return []
        
        # Get all image files from the directory
        all_images = []
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))
        
        # Get excluded images (labeled as not_a_crow or not_sure)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_path FROM image_labels 
            WHERE label IN ('not_a_crow', 'not_sure')
        """)
        excluded_paths = {row[0] for row in cursor.fetchall()}
        
        # Return images that are not excluded
        suitable = [path for path in all_images if path not in excluded_paths]
        return suitable
        
    except Exception as e:
        logger.error(f"Error getting training suitable images: {e}")
        return []

def is_image_training_suitable(image_path):
    """Check if a specific image is suitable for training."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT label, is_training_data 
            FROM image_labels 
            WHERE image_path = ?
        ''', (image_path,))
        
        result = cursor.fetchone()
        if result:
            label, is_training_data = result
            # Suitable if marked as training data and not labeled as 'not_a_crow' or 'not_sure'
            return is_training_data and label not in ('not_a_crow', 'not_sure')
        else:
            # If not labeled, assume it's suitable (for backward compatibility)
            return True
            
    except Exception as e:
        logger.error(f"Error checking if image is training suitable: {e}")
        return True  # Default to suitable if there's an error
    finally:
        if conn:
            conn.close()

def get_crow_videos(crow_id):
    """Get all videos where a specific crow appears."""
    conn = get_connection()
    c = conn.cursor()
    
    # Verify crow exists
    c.execute('SELECT id FROM crows WHERE id = ?', (crow_id,))
    if not c.fetchone():
        raise ValueError(f"Crow ID {crow_id} does not exist")
    
    # Try to get videos from embeddings table first
    c.execute('''
        SELECT DISTINCT video_path, COUNT(*) as sighting_count,
               MIN(timestamp) as first_seen_in_video,
               MAX(timestamp) as last_seen_in_video
        FROM crow_embeddings
        WHERE crow_id = ? AND video_path IS NOT NULL
        GROUP BY video_path
        ORDER BY first_seen_in_video
    ''', (crow_id,))
    
    rows = c.fetchall()
    conn.close()
    
    # If we found embeddings data, return it
    if rows:
        return [{
            'video_path': row[0],
            'sighting_count': row[1],
            'first_seen': row[2],
            'last_seen': row[3]
        } for row in rows]
    
    # Fallback: scan crop directory directly for video names
    # This handles cases where crop images exist but no embeddings in database
    from pathlib import Path
    import os
    
    crop_dir = Path("crow_crops") / str(crow_id)
    if not crop_dir.exists():
        return []
    
    # Extract video names from image filenames
    video_counts = {}
    for image_file in crop_dir.glob("*.jpg"):
        filename = image_file.name
        # Expected format: {video_name}_frame{frame_num}_{detection_num}.jpg
        if '_frame' in filename:
            video_part = filename.split('_frame')[0]
            # Try to reconstruct original video name (add common extensions)
            for ext in ['.mp4', '.mov', '.avi', '.MOV', '.MP4']:
                video_name = video_part + ext
                video_counts[video_name] = video_counts.get(video_name, 0) + 1
    
    # Return video data based on crop directory contents
    return [{
        'video_path': video_path,
        'sighting_count': count,
        'first_seen': None,  # Not available from crop directory
        'last_seen': None    # Not available from crop directory
    } for video_path, count in video_counts.items()]

def get_first_crow_image(crow_id):
    """Get the path to the first crop image for a specific crow."""
    import os
    from pathlib import Path
    
    # Look directly in the crop directory for this crow
    crop_dir = Path("crow_crops") / str(crow_id)
    if crop_dir.exists():
        # Find the first image chronologically by filename
        image_files = list(crop_dir.glob("*.jpg"))
        if image_files:
            # Sort by filename (which should be chronological)
            image_files.sort()
            return str(image_files[0])
    
    return None

def get_crow_images_from_video(crow_id, video_path):
    """Get all crop images for a specific crow from a specific video."""
    import os
    from pathlib import Path
    
    # Look for crop images directly in the crop directory
    crop_dir = Path("crow_crops") / str(crow_id)
    if not crop_dir.exists():
        return []
    
    # Extract video name from path (remove extension and clean up)
    video_name = Path(video_path).stem  # Gets filename without extension
    
    # Find images that match this video
    matching_images = []
    for image_file in crop_dir.glob("*.jpg"):
        filename = image_file.name
        # Check if the video name (without extension) appears in the filename
        if video_name.replace('.', '_').replace(' ', '_') in filename:
            matching_images.append(str(image_file))
    
    # If we found matching images, return them
    if matching_images:
        matching_images.sort()
        return matching_images
    
    # Fallback: if no specific video match, return some images from this crow
    # This happens when video names don't match exactly
    all_images = list(crop_dir.glob("*.jpg"))
    if all_images:
        all_images.sort()
        return all_images[:10]  # Return first 10 images
    
    return []

def reassign_crow_embeddings(from_crow_id, to_crow_id, embedding_ids=None):
    """
    Reassign embeddings from one crow to another.
    
    Args:
        from_crow_id: ID of the crow to move embeddings from
        to_crow_id: ID of the crow to move embeddings to  
        embedding_ids: List of specific embedding IDs to move. If None, moves all embeddings.
    
    Returns:
        int: Number of embeddings moved
    """
    conn = None
    try:
        conn = get_connection()
        conn.execute("BEGIN TRANSACTION")
        cursor = conn.cursor()
        
        # Verify both crows exist
        cursor.execute('SELECT id FROM crows WHERE id IN (?, ?)', (from_crow_id, to_crow_id))
        existing_crows = cursor.fetchall()
        if len(existing_crows) != 2:
            raise ValueError(f"One or both crow IDs do not exist: {from_crow_id}, {to_crow_id}")
        
        # Get embeddings to move
        if embedding_ids:
            # Move specific embeddings
            placeholders = ','.join(['?'] * len(embedding_ids))
            cursor.execute(f'''
                SELECT id FROM crow_embeddings 
                WHERE crow_id = ? AND id IN ({placeholders})
            ''', [from_crow_id] + embedding_ids)
        else:
            # Move all embeddings from the crow
            cursor.execute('''
                SELECT id FROM crow_embeddings 
                WHERE crow_id = ?
            ''', (from_crow_id,))
        
        embeddings_to_move = [row[0] for row in cursor.fetchall()]
        
        if not embeddings_to_move:
            logger.warning(f"No embeddings found to move from crow {from_crow_id}")
            return 0
        
        # Update the embeddings
        placeholders = ','.join(['?'] * len(embeddings_to_move))
        cursor.execute(f'''
            UPDATE crow_embeddings 
            SET crow_id = ? 
            WHERE id IN ({placeholders})
        ''', [to_crow_id] + embeddings_to_move)
        
        moved_count = cursor.rowcount
        
        # Update sighting counts
        # Decrease from_crow sightings
        cursor.execute('''
            UPDATE crows 
            SET total_sightings = total_sightings - ? 
            WHERE id = ?
        ''', (moved_count, from_crow_id))
        
        # Increase to_crow sightings  
        cursor.execute('''
            UPDATE crows 
            SET total_sightings = total_sightings + ? 
            WHERE id = ?
        ''', (moved_count, to_crow_id))
        
        # Update timestamps for to_crow
        cursor.execute('''
            SELECT MIN(timestamp), MAX(timestamp) 
            FROM crow_embeddings 
            WHERE crow_id = ?
        ''', (to_crow_id,))
        
        times = cursor.fetchone()
        if times and times[0] and times[1]:
            cursor.execute('''
                UPDATE crows 
                SET first_seen = ?, last_seen = ? 
                WHERE id = ?
            ''', (times[0], times[1], to_crow_id))
        
        # Check if from_crow has any remaining embeddings
        cursor.execute('SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?', (from_crow_id,))
        remaining_count = cursor.fetchone()[0]
        
        if remaining_count == 0:
            # Update from_crow to have 0 sightings
            cursor.execute('''
                UPDATE crows 
                SET total_sightings = 0 
                WHERE id = ?
            ''', (from_crow_id,))
        else:
            # Update from_crow timestamps
            cursor.execute('''
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM crow_embeddings 
                WHERE crow_id = ?
            ''', (from_crow_id,))
            
            times = cursor.fetchone()
            if times and times[0] and times[1]:
                cursor.execute('''
                    UPDATE crows 
                    SET first_seen = ?, last_seen = ? 
                    WHERE id = ?
                ''', (times[0], times[1], from_crow_id))
        
        conn.commit()
        logger.info(f"Successfully reassigned {moved_count} embeddings from crow {from_crow_id} to crow {to_crow_id}")
        return moved_count
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error reassigning crow embeddings: {e}")
        raise
    finally:
        if conn:
            conn.close()

def create_new_crow_from_embeddings(from_crow_id, embedding_ids, new_crow_name=None):
    """
    Create a new crow and move specific embeddings to it.
    
    Args:
        from_crow_id: ID of the crow to move embeddings from
        embedding_ids: List of embedding IDs to move to the new crow
        new_crow_name: Optional name for the new crow
    
    Returns:
        int: ID of the newly created crow
    """
    conn = None
    try:
        conn = get_connection()
        conn.execute("BEGIN TRANSACTION")
        cursor = conn.cursor()
        
        # Verify from_crow exists
        cursor.execute('SELECT id FROM crows WHERE id = ?', (from_crow_id,))
        if not cursor.fetchone():
            raise ValueError(f"Crow ID {from_crow_id} does not exist")
        
        # Verify embeddings exist and belong to from_crow
        placeholders = ','.join(['?'] * len(embedding_ids))
        cursor.execute(f'''
            SELECT id FROM crow_embeddings 
            WHERE crow_id = ? AND id IN ({placeholders})
        ''', [from_crow_id] + embedding_ids)
        
        valid_embeddings = [row[0] for row in cursor.fetchall()]
        if len(valid_embeddings) != len(embedding_ids):
            raise ValueError("Some embedding IDs don't exist or don't belong to the specified crow")
        
        # Get timestamp range for the new crow
        cursor.execute(f'''
            SELECT MIN(timestamp), MAX(timestamp) 
            FROM crow_embeddings 
            WHERE id IN ({placeholders})
        ''', embedding_ids)
        
        times = cursor.fetchone()
        first_seen = times[0] if times else None
        last_seen = times[1] if times else None
        
        # Create new crow
        cursor.execute('''
            INSERT INTO crows (name, first_seen, last_seen, total_sightings) 
            VALUES (?, ?, ?, ?)
        ''', (new_crow_name, first_seen, last_seen, len(embedding_ids)))
        
        new_crow_id = cursor.lastrowid
        
        # Move embeddings to new crow
        cursor.execute(f'''
            UPDATE crow_embeddings 
            SET crow_id = ? 
            WHERE id IN ({placeholders})
        ''', [new_crow_id] + embedding_ids)
        
        # Update original crow's sighting count and timestamps
        cursor.execute('''
            UPDATE crows 
            SET total_sightings = total_sightings - ? 
            WHERE id = ?
        ''', (len(embedding_ids), from_crow_id))
        
        # Check if original crow has remaining embeddings
        cursor.execute('SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?', (from_crow_id,))
        remaining_count = cursor.fetchone()[0]
        
        if remaining_count > 0:
            # Update original crow timestamps
            cursor.execute('''
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM crow_embeddings 
                WHERE crow_id = ?
            ''', (from_crow_id,))
            
            times = cursor.fetchone()
            if times and times[0] and times[1]:
                cursor.execute('''
                    UPDATE crows 
                    SET first_seen = ?, last_seen = ? 
                    WHERE id = ?
                ''', (times[0], times[1], from_crow_id))
        else:
            # Original crow has no embeddings left
            cursor.execute('''
                UPDATE crows 
                SET total_sightings = 0 
                WHERE id = ?
            ''', (from_crow_id,))
        
        conn.commit()
        logger.info(f"Created new crow {new_crow_id} with {len(embedding_ids)} embeddings from crow {from_crow_id}")
        return new_crow_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error creating new crow from embeddings: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_embedding_ids_by_image_paths(image_paths):
    """
    Get embedding IDs corresponding to image paths.
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        dict: Mapping of image_path -> embedding_id
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        result = {}
        
        for img_path in image_paths:
            # Extract video name and frame number from image path
            # Expected format: crow_crops/{crow_id}/{video_name}_frame{frame_num}_{detection_num}.jpg
            filename = os.path.basename(img_path)
            
            # Parse filename to extract video and frame info
            if '_frame' in filename:
                parts = filename.split('_frame')
                if len(parts) >= 2:
                    video_part = parts[0]
                    frame_part = parts[1].split('_')[0]  # Get frame number before detection number
                    
                    try:
                        frame_number = int(frame_part)
                        
                        # Find matching embedding
                        cursor.execute('''
                            SELECT id FROM crow_embeddings 
                            WHERE video_path LIKE ? AND frame_number = ?
                            ORDER BY timestamp DESC
                            LIMIT 1
                        ''', (f'%{video_part}%', frame_number))
                        
                        row = cursor.fetchone()
                        if row:
                            result[img_path] = row[0]
                        else:
                            logger.warning(f"No embedding found for image: {img_path}")
                            
                    except ValueError:
                        logger.warning(f"Could not parse frame number from: {filename}")
            else:
                logger.warning(f"Unexpected filename format: {filename}")
                
        return result
        
    except Exception as e:
        logger.error(f"Error getting embedding IDs by image paths: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def delete_crow_embeddings(embedding_ids):
    """
    Delete specific crow embeddings from the database.
    
    Args:
        embedding_ids: List of embedding IDs to delete
    
    Returns:
        int: Number of embeddings deleted
    """
    conn = None
    try:
        conn = get_connection()
        conn.execute("BEGIN TRANSACTION")
        cursor = conn.cursor()
        
        if not embedding_ids:
            return 0
            
        # Get crow IDs that will be affected
        placeholders = ','.join(['?'] * len(embedding_ids))
        cursor.execute(f'''
            SELECT DISTINCT crow_id FROM crow_embeddings 
            WHERE id IN ({placeholders})
        ''', embedding_ids)
        
        affected_crow_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete the embeddings
        cursor.execute(f'''
            DELETE FROM crow_embeddings 
            WHERE id IN ({placeholders})
        ''', embedding_ids)
        
        deleted_count = cursor.rowcount
        
        # Update affected crows' statistics
        for crow_id in affected_crow_ids:
            # Update sighting count
            cursor.execute('''
                SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?
            ''', (crow_id,))
            
            remaining_count = cursor.fetchone()[0]
            
            if remaining_count > 0:
                # Update timestamps and count
                cursor.execute('''
                    SELECT MIN(timestamp), MAX(timestamp) 
                    FROM crow_embeddings 
                    WHERE crow_id = ?
                ''', (crow_id,))
                
                times = cursor.fetchone()
                if times and times[0] and times[1]:
                    cursor.execute('''
                        UPDATE crows 
                        SET first_seen = ?, last_seen = ?, total_sightings = ? 
                        WHERE id = ?
                    ''', (times[0], times[1], remaining_count, crow_id))
            else:
                # No embeddings left for this crow
                cursor.execute('''
                    UPDATE crows 
                    SET total_sightings = 0 
                    WHERE id = ?
                ''', (crow_id,))
        
        conn.commit()
        logger.info(f"Deleted {deleted_count} embeddings")
        return deleted_count
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error deleting crow embeddings: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Initialize database on module import
initialize_database() 