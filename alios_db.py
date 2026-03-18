import sqlite3
import json
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "Artifacts")
DB_PATH = os.path.join(BASE_DIR, "alios_meta.db")

# Ensure the artifacts folder exists
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR)

def init_db():
    """Initializes the SQLite database with a flexible schema."""
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()

def register_run(run_id, algo, state_repr, config_dict, q_table):
    """
    Standardizes a training run and saves it to the Alios Registry.
    
    Args:
        run_id (str): Unique name for the experiment (e.g., 'q_learning_v1')
        algo (str): Name of the algorithm (e.g., 'SARSA')
        state_repr (str): KEY for the UI to know how to decode (e.g., '3x3_base3')
        config_dict (dict): Any hacky config settings you want to keep
        q_table (np.ndarray): The final (N, 4) Q-table
    """
    # 1. Save the Q-table artifact
    policy_filename = f"{run_id}_policy.npy"
    policy_path = os.path.join(ARTIFACT_DIR, policy_filename)
    np.save(policy_path, q_table)
    
    # 2. Convert config dict to JSON string
    config_json = json.dumps(config_dict)
    
    # 3. Update the SQLite Database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Use REPLACE to allow overwriting a run if you re-run a prototype
    cursor.execute('''
        INSERT OR REPLACE INTO runs (run_id, algo, state_repr, config_json, policy_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (run_id, algo, state_repr, config_json, policy_path, datetime.now()))
    
    conn.commit()
    conn.close()
    print(f"✅ Run '{run_id}' successfully registered in Alios.")

# Initialize the DB immediately when this script is imported
init_db()