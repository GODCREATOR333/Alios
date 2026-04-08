# tests/test_db.py
import sqlite3
import numpy as np
import db_manager

def test_database_registration_and_retrieval(tmp_path, monkeypatch):
    """Safely tests writing and reading from the DB using a temporary directory."""
    
    # 1. ARRANGE: Hijack the db_manager's paths to point to our temp folder!
    temp_db_dir = tmp_path / "Database"
    temp_art_dir = tmp_path / "Artifacts"
    temp_db_path = temp_db_dir / "Alios_meta.db"
    
    monkeypatch.setattr(db_manager, "DB_DIR", str(temp_db_dir))
    monkeypatch.setattr(db_manager, "ARTIFACT_DIR", str(temp_art_dir))
    monkeypatch.setattr(db_manager, "DB_PATH", str(temp_db_path))
    monkeypatch.setattr(db_manager, "BASE_DIR", str(tmp_path))
    
    # Force initialize the fake DB
    db_manager.init_db()
    
    # 2. ACT: Create a fake Q-table and save it
    fake_q = np.zeros((256, 4))
    db_manager.register_run(
        run_id="test_run_001",
        algo="Q-Learning",
        state_repr="mdp",
        config_dict={"learning_rate": 0.05},
        q_table=fake_q
    )
    
    # 3. ASSERT: Did it save? Can we get it back?
    runs = db_manager.get_all_runs()
    assert "test_run_001" in runs
    
    details = db_manager.get_run_details("test_run_001")
    assert details is not None
    assert details["algo"] == "Q-Learning"
    assert details["config"]["learning_rate"] == 0.05
    
    # Did it physically save the numpy file?
    loaded_q = np.load(details["path"])
    assert loaded_q.shape == (256, 4)