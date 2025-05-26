import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_database_with_crops():
    """
    Sync the database with existing crop directories.
    Creates crow entries for all numeric directories found in crow_crops.
    """
    try:
        # Get all numeric directories in crow_crops
        crop_base = Path("crow_crops")
        if not crop_base.exists():
            logger.error("crow_crops directory does not exist")
            return
        
        # Find all numeric subdirectories
        numeric_dirs = []
        for subdir in crop_base.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                numeric_dirs.append(int(subdir.name))
        
        numeric_dirs.sort()
        logger.info(f"Found {len(numeric_dirs)} crop directories: {numeric_dirs[:10]}..." if len(numeric_dirs) > 10 else f"Found {len(numeric_dirs)} crop directories: {numeric_dirs}")
        
        # Get existing crows from database
        existing_crows = db.get_all_crows()
        existing_ids = {crow['id'] for crow in existing_crows}
        logger.info(f"Existing crows in database: {existing_ids}")
        
        # Create database entries for missing directories
        conn = db.get_connection()
        cursor = conn.cursor()
        
        created_count = 0
        for crow_id in numeric_dirs:
            if crow_id not in existing_ids:
                # Count images in this directory
                crop_dir = crop_base / str(crow_id)
                image_count = len(list(crop_dir.glob("*.jpg")))
                
                if image_count > 0:
                    # Create crow entry
                    cursor.execute('''
                        INSERT INTO crows (id, name, total_sightings, first_seen, last_seen)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ''', (crow_id, f"Crow {crow_id}", image_count))
                    
                    created_count += 1
                    logger.info(f"Created crow {crow_id} with {image_count} images")
                else:
                    logger.warning(f"Directory {crow_id} has no images, skipping")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully created {created_count} new crow entries")
        
        # Show final stats
        all_crows = db.get_all_crows()
        logger.info(f"Database now contains {len(all_crows)} crows")
        
        return created_count
        
    except Exception as e:
        logger.error(f"Error syncing database: {e}")
        raise

if __name__ == "__main__":
    print("Syncing database with existing crop directories...")
    created = sync_database_with_crops()
    print(f"Created {created} new crow entries in the database")
    print("You can now restart the suspect lineup tool to see all crows!") 