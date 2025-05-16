import sqlite3
import numpy as np
import os

DB_PATH = 'crow_embeddings.db'

# Ensure the database and table exist
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS crows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding BLOB NOT NULL,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()
conn.close()

def save_crow_embedding(embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    embedding_blob = embedding.astype(np.float32).tobytes()
    c.execute('INSERT INTO crows (embedding) VALUES (?)', (embedding_blob,))
    crow_id = c.lastrowid
    conn.commit()
    conn.close()
    return crow_id

def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, embedding FROM crows')
    rows = c.fetchall()
    conn.close()
    embeddings = []
    for row in rows:
        crow_id, emb_blob = row
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        embeddings.append((crow_id, emb))
    return embeddings

def update_last_seen(crow_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE crows SET last_seen = CURRENT_TIMESTAMP WHERE id = ?', (crow_id,))
    conn.commit()
    conn.close() 