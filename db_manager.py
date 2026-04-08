# db_manager.py
import sqlite3
import json
import numpy as np
import os
from datetime import datetime

# --- Robust Path Configuration ---
# BASE_DIR is the directory where db_manager.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "Artifacts")
DB_DIR = os.path.join(BASE_DIR, "Database")
DB_PATH = os.path.join(DB_DIR, "Alios_meta.db")

def _ensure_directories():
    """Internal helper to ensure required directories exist."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

def init_db():
    """Initializes the SQLite database with a flexible schema."""
    _ensure_directories()
    
    # Using 'with' ensures the connection commits automatically and closes safely
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                algo TEXT,
                state_repr TEXT,
                config_json TEXT,
                policy_path TEXT,
                timestamp DATETIME
            )
        ''')

def register_run(run_id, algo, state_repr, config_dict, q_table):
    """
    Standardizes a training run and saves it to the Alios Registry.
    """
    _ensure_directories()
    
    # 1. Save the Q-table artifact
    policy_filename = f"{run_id}_policy.npy"
    full_save_path = os.path.join(ARTIFACT_DIR, policy_filename)
    np.save(full_save_path, q_table)
    
    # 2. Store RELATIVE path in DB (makes project portable)
    # E.g., "Artifacts/DMP_REACTIVE_SWITCHER_V1_policy.npy"
    relative_policy_path = os.path.join("Artifacts", policy_filename)
    
    # 3. Convert config dict to JSON string
    config_json = json.dumps(config_dict)
    
    # 4. Update Database safely
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO runs (run_id, algo, state_repr, config_json, policy_path, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (run_id, algo, state_repr, config_json, relative_policy_path, datetime.now()))
        
    print(f"✅ Run '{run_id}' successfully registered in Alios.")

def get_all_runs():
    """Returns a list of all run_ids from the DB."""
    if not os.path.exists(DB_PATH): 
        return []
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM runs ORDER BY timestamp DESC")
        return [row[0] for row in cursor.fetchall()]

def get_run_details(run_id):
    """Returns the config and absolute policy path for a specific run."""
    if not os.path.exists(DB_PATH):
        return None

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT algo, state_repr, config_json, policy_path FROM runs WHERE run_id=?", (run_id,))
        row = cursor.fetchone()
        
    if row:
        # Convert relative path back to absolute path for the Engine to read safely
        relative_path = row[3]
        absolute_path = os.path.join(BASE_DIR, relative_path)
        
        return {
            "algo": row[0],
            "state_repr": row[1],
            "config": json.loads(row[2]),
            "path": absolute_path
        }
    return None

# We initialize the DB safely at the bottom, ensuring directories exist.
init_db()