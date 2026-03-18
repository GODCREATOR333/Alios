from alios_db import register_run
import numpy as np

# Create a fake Q-table for MDP (256 states, 4 actions)
fake_q = np.random.rand(256, 4)

register_run(
    run_id="PROTOTYPE_001",
    algo="Q-Learning",
    state_repr="mdp",
    config_dict={"gamma": 0.99, "notes": "Testing the Alios handshake"},
    q_table=fake_q
)