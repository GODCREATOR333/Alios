# register_dummy.py
import numpy as np
from alios_db import register_run

# 1. Generate a random Q-table
# 26244 states (3^8 * 4), 4 actions
random_q = np.random.uniform(-10, 10, (26244, 4))

# 2. Register it in the database
register_run(
    run_id="RANDOM_AGENT_TEST",
    algo="Random Baseline",
    state_repr="3x3_base3_compass",
    config_dict={
        "gamma": 0.99,
        "notes": "Testing Alios visualization with a random policy",
        "status": "dummy"
    },
    q_table=random_q
)

print("Successfully registered RANDOM_AGENT_TEST")