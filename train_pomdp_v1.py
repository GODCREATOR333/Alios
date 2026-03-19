import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import os
import sys
import time

# --- 1. Link to Alios Registry ---
alios_path = "../Alios" # Update this to your actual path
print(alios_path)
sys.path.append(alios_path)
from alios_db import register_run

# --- 2. JAX Environment Setup (The logic we perfected) ---

@jax.jit
def get_state(maze, r, c):
    # Base-3 Encoding (0=path, 1=wall, 2=goal)
    maze_with_goal = maze.at[15, 15].set(2)
    padded = jnp.pad(maze_with_goal, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3))
    flat = window.flatten()
    neighbors = jnp.concatenate([flat[:4], flat[5:]])
    powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1])
    local_id = jnp.sum(neighbors * powers)
    dr = jnp.where(r < 15, 1, 0)
    dc = jnp.where(c < 15, 1, 0)
    compass_id = dr * 2 + dc
    return (local_id * 4 + compass_id).astype(jnp.int32)

@jax.jit
def step(maze, r, c, last_r, last_c, action):
    dr_array = jnp.array([-1, 1, 0, 0])
    dc_array = jnp.array([0, 0, -1, 1])
    nr, nc = r + dr_array[action], c + dc_array[action]
    
    # Physics
    invalid = (nr<0)|(nr>=16)|(nc<0)|(nc>=16) | (maze[jnp.clip(nr,0,15), jnp.clip(nc,0,15)]==1)
    fr, fc = jnp.where(invalid, r, nr), jnp.where(invalid, c, nc)
    
    # Rewards
    is_goal = (fr==15)&(fc==15)
    dist_reward = jnp.where((jnp.abs(fr-15)+jnp.abs(fc-15)) < (jnp.abs(r-15)+jnp.abs(c-15)), 0.5, -0.5)
    uturn = (fr==last_r)&(fc==last_c)&(~invalid)
    
    reward = jnp.where(is_goal, 100.0, jnp.where(invalid, -10.0, -1.0)) + dist_reward + jnp.where(uturn, -2.0, 0.0)
    return fr, fc, reward, is_goal

# --- 3. The JAX Training Loop ---

@jax.jit
def play_episode(q_table, maze, epsilon, key):
    def cond_fun(b): return (~b[6]) & (b[5] < 500) # not done and steps < 500
    def body_fun(b):
        q, r, c, lr, lc, steps, done, k = b
        s = get_state(maze, r, c)
        k, k1, k2 = jrandom.split(k, 3)
        a = jnp.where(jrandom.uniform(k1) < epsilon, jrandom.randint(k2, (), 0, 4), jnp.argmax(q[s]))
        nr, nc, rew, n_done = step(maze, r, c, lr, lc, a)
        ns = get_state(maze, nr, nc)
        target = rew + 0.99 * jnp.where(n_done, 0.0, jnp.max(q[ns]))
        q = q.at[s, a].set(q[s, a] + 1.0 * (target - q[s, a])) # Alpha = 1.0
        return q, nr, nc, r, c, steps + 1, n_done, k

    init = (q_table, 0, 0, 0, 0, 0, False, key)
    res = jax.lax.while_loop(cond_fun, body_fun, init)
    return res[0], res[7] # Return Q and Key

# --- 4. Main Execution ---

# Load the large training set
train_data = np.load("data_jax/N16_P0100_train_solvable.npy")
train_jax = jnp.array(train_data)

q_table = jnp.full((26244, 4), 150.0) # Optimistic init
key = jrandom.PRNGKey(42)
epsilon, decay = 1.0, 0.9998

print(f"Starting Training on {len(train_data)} mazes...")
start = time.time()

# Train on 20,000 episodes, picking random mazes from the 4,500 available
for ep in range(20000):
    maze_idx = jrandom.randint(key, (), 0, 4500)
    epsilon = max(0.01, epsilon * decay)
    q_table, key = play_episode(q_table, train_jax[maze_idx], epsilon, key)
    
    if ep % 1000 == 0:
        print(f"Episode {ep} | Epsilon: {epsilon:.2f}")

# --- 5. Register with Alios ---
register_run(
    run_id="POMDP_Base3_20k_Alpha1",
    algo="Q-Learning",
    state_repr="3x3_base3_compass",
    config_dict={
        "episodes": 20000,
        "train_set": "N16_P0100_train_solvable",
        "alpha": 1.0,
        "gamma": 0.99,
        "initial_q": 150.0,
        "shaping": "dist + uturn_penalty"
    },
    q_table=np.array(q_table)
)

print(f"Total training time: {time.time()-start:.2f}s")