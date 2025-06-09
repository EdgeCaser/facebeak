import pytest
import sqlite3
import numpy as np
from pathlib import Path

# Add project root to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import db # Assuming direct import works for db.py functions

@pytest.fixture
def memory_db_connection():
    """Fixture to create an in-memory SQLite database and initialize schema."""
    conn = sqlite3.connect(':memory:')
    db.initialize_database(conn) # Assumes db.py has this function
    yield conn
    conn.close()

def test_save_and_retrieve_crow_embedding(memory_db_connection):
    """
    Tests saving a new crow embedding with all metadata and retrieving it.
    Also tests that multiple embeddings for the same crow_id can be saved.
    """
    conn = memory_db_connection

    # Sample data for the first embedding
    embedding1 = np.random.rand(128).astype(np.float32) # Example 128-dim embedding
    video_path1 = "/videos/test_video_01.mp4"
    frame_num1 = 101
    crow_id1 = "crow_test_A"
    confidence1 = 0.95
    bbox_x1, bbox_y1, bbox_w1, bbox_h1 = 10, 20, 30, 40

    db.save_crow_embedding(
        embedding=embedding1,
        video_path=video_path1,
        frame_number=frame_num1,
        crow_id=crow_id1,
        confidence=confidence1,
        bbox_x=bbox_x1,
        bbox_y=bbox_y1,
        bbox_w=bbox_w1,
        bbox_h=bbox_h1,
        db_connection=conn # Pass the connection
    )

    # Verify the first embedding
    cursor = conn.cursor()
    cursor.execute("""
        SELECT embedding_data, video_path, frame_number, crow_id, confidence,
               bbox_x, bbox_y, bbox_w, bbox_h
        FROM crow_embeddings
        WHERE video_path = ? AND frame_number = ? AND crow_id = ?
    """, (video_path1, frame_num1, crow_id1))
    row = cursor.fetchone()

    assert row is not None, "First embedding not found in database."
    retrieved_embedding1 = np.frombuffer(row[0], dtype=np.float32)
    assert np.array_equal(retrieved_embedding1, embedding1), "Embedding data mismatch for first save."
    assert row[1] == video_path1
    assert row[2] == frame_num1
    assert row[3] == crow_id1
    assert row[4] == confidence1
    assert row[5] == bbox_x1
    assert row[6] == bbox_y1
    assert row[7] == bbox_w1
    assert row[8] == bbox_h1

    # Sample data for a second embedding (same crow_id, different frame)
    embedding2 = np.random.rand(128).astype(np.float32)
    video_path2 = "/videos/test_video_01.mp4" # Can be same video
    frame_num2 = 202
    # crow_id1 remains the same
    confidence2 = 0.88
    bbox_x2, bbox_y2, bbox_w2, bbox_h2 = 50, 60, 70, 80

    db.save_crow_embedding(
        embedding=embedding2,
        video_path=video_path2,
        frame_number=frame_num2,
        crow_id=crow_id1, # Same crow_id
        confidence=confidence2,
        bbox_x=bbox_x2,
        bbox_y=bbox_y2,
        bbox_w=bbox_w2,
        bbox_h=bbox_h2,
        db_connection=conn
    )

    # Verify the second embedding
    cursor.execute("""
        SELECT embedding_data, video_path, frame_number, confidence
        FROM crow_embeddings
        WHERE video_path = ? AND frame_number = ? AND crow_id = ?
    """, (video_path2, frame_num2, crow_id1))
    row2 = cursor.fetchone()

    assert row2 is not None, "Second embedding for the same crow_id not found."
    retrieved_embedding2 = np.frombuffer(row2[0], dtype=np.float32)
    assert np.array_equal(retrieved_embedding2, embedding2), "Embedding data mismatch for second save."
    assert row2[1] == video_path2
    assert row2[2] == frame_num2
    assert row2[3] == confidence2

    # Verify there are two distinct entries for this crow_id now
    cursor.execute("SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?", (crow_id1,))
    count = cursor.fetchone()[0]
    assert count == 2, "Should have two embedding entries for the same crow_id but different contexts."
