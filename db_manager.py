import sqlite3
import json
import os

DB_PATH = "Alios_meta.db"

def get_all_runs():
    """Returns a list of all run_ids from the DB."""
    if not os.path.exists(DB_PATH): return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT run_id FROM runs ORDER BY timestamp DESC")
    runs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return runs

def get_run_details(run_id):
    """Returns the config and policy path for a specific run."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT algo, state_repr, config_json, policy_path FROM runs WHERE run_id=?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "algo": row[0],
            "state_repr": row[1],
            "config": json.loads(row[2]),
            "path": row[3]
        }
    return None