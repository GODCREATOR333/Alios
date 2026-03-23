import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import numpy as np
import matplotlib.pyplot as plt
import os
from alios_db import register_run

# --- Parameters ---
GRID_SIZE = 16
N_ACTIONS = 4
START = jnp.array([0, 0], dtype=jnp.int32)
GOAL = jnp.array([GRID_SIZE - 1, GRID_SIZE - 1], dtype=jnp.int32)
HALF_W = 1 # 3x3 window

ACTION_DELTAS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)

# --- 1. State Encoding (4D Tuple: 256-state window) ---
@jit
def get_state_4d(maze, pos):
    row, col = pos
    padded_maze = jnp.pad(maze, HALF_W, constant_values=1)
    window = lax.dynamic_slice(padded_maze, (row, col), (3, 3)).flatten()
    
    # 9-bit window (512 combinations)
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    window_int = jnp.sum(window * powers)
    
    dr_goal = jnp.sign(GOAL[0] - row) + 1
    dc_goal = jnp.sign(GOAL[1] - col) + 1
    return window_int, dr_goal, dc_goal

# --- 2. Environment Step ---
@jit
def step(maze, pos, action):
    delta = ACTION_DELTAS[action]
    new_pos = pos + delta
    row, col = new_pos
    
    out_of_bounds = (row < 0) | (row >= GRID_SIZE) | (col < 0) | (col >= GRID_SIZE)
    safe_row, safe_col = jnp.clip(row, 0, GRID_SIZE-1), jnp.clip(col, 0, GRID_SIZE-1)
    hit_wall = out_of_bounds | (maze[safe_row, safe_col] == 1)
    
    is_goal = (row == GOAL[0]) & (col == GOAL[1]) & ~hit_wall
    actual_new_pos = jnp.where(hit_wall, pos, new_pos)
    
    old_dist = jnp.abs(pos[0] - GOAL[0]) + jnp.abs(pos[1] - GOAL[1])
    new_dist = jnp.abs(actual_new_pos[0] - GOAL[0]) + jnp.abs(actual_new_pos[1] - GOAL[1])
    progress_reward = 0.50 * (old_dist - new_dist)
    
    reward = jnp.where(hit_wall, -10.0, 
                jnp.where(is_goal, 100.0, -0.20 + progress_reward))
    
    done = is_goal | hit_wall  # <--- Terminate on wall for better training signal
    return actual_new_pos, reward, done

# --- 3. Action Selection (Now uses 4D Indexing) ---
@jit
def select_action(q_table, w, dr, dc, epsilon, pos, rng):
    rng_eps, rng_act, rng_noise = jax.random.split(rng, 3)
    
    q_values = q_table[w, dr, dc]
    
    # Add tiny random noise to break ties randomly instead of always picking 'UP'
    noise = jax.random.uniform(rng_noise, shape=(4,), minval=0, maxval=1e-7)
    
    dr_goal = jnp.sign(GOAL[0] - pos[0])
    dc_goal = jnp.sign(GOAL[1] - pos[1])
    bonus = jnp.array([
        (ACTION_DELTAS[0][0] == dr_goal) * 1e-5,
        (ACTION_DELTAS[1][0] == dr_goal) * 1e-5,
        (ACTION_DELTAS[2][1] == dc_goal) * 1e-5,
        (ACTION_DELTAS[3][1] == dc_goal) * 1e-5,
    ])
    
    best_action = jnp.argmax(q_values + bonus + noise)
    random_action = jax.random.randint(rng_act, shape=(), minval=0, maxval=N_ACTIONS)
    
    return jnp.where(jax.random.uniform(rng_eps) < epsilon, random_action, best_action)

# --- 4. The Training Engine ---
@jit
def train_episode(carry, episode_idx):
    q_table, rng, mazes, epsilon_decay, epsilon_end = carry
    
    rng, rng_maze, rng_action = jax.random.split(rng, 3)
    maze_idx = jax.random.randint(rng_maze, (), 0, mazes.shape[0])
    maze = mazes[maze_idx]
    
    epsilon = jnp.maximum(epsilon_end, 1.0 * (epsilon_decay ** episode_idx))
    
    init_loop_state = (START, q_table, 0.0, 0, False, rng_action)
    max_steps = GRID_SIZE * GRID_SIZE * 8

    def cond_fun(loop_state):
        _, _, _, steps, done, _ = loop_state
        return jnp.logical_not(done) & (steps < max_steps)

    def body_fun(loop_state):
        pos, current_q, total_reward, steps, done, key = loop_state
        key, step_key = jax.random.split(key)
        
        w, dr, dc = get_state_4d(maze, pos)
        action = select_action(current_q, w, dr, dc, epsilon, pos, step_key)
        
        new_pos, reward, done_step = step(maze, pos, action)
        new_w, new_dr, new_dc = get_state_4d(maze, new_pos)
        
        alpha, gamma = 0.2, 0.99
        
        target_q = jnp.where(
            done_step, reward,
            reward + gamma * jnp.max(current_q[new_w, new_dr, new_dc])
        )
        
        td_error = target_q - current_q[w, dr, dc, action]
        
        # 4D matrix update
        updated_q = current_q.at[w, dr, dc, action].add(alpha * td_error)
        
        return (new_pos, updated_q, total_reward + reward, steps + 1, done_step, key)

    final_loop_state = lax.while_loop(cond_fun, body_fun, init_loop_state)
    _, final_q_table, ep_reward, _, _, _ = final_loop_state
    
    return (final_q_table, rng, mazes, epsilon_decay, epsilon_end), ep_reward

def train_jax(mazes, n_episodes=50000):
    print("🚀 Compiling and Training...")
    # Initialize 4D Q-Table: (256 windows, 3 row_dirs, 3 col_dirs, 4 actions)
    q_table = jnp.zeros((512, 3, 3, 4))
    rng = jax.random.PRNGKey(42)
    epsilon_end, epsilon_decay = 0.01, (0.01 / 1.0) ** (1 / n_episodes)
    
    carry = (q_table, rng, jnp.array(mazes), epsilon_decay, epsilon_end)
    final_carry, rewards = lax.scan(train_episode, carry, jnp.arange(n_episodes))
    
    return final_carry[0], np.array(rewards)

# --- 5. BATCH EVALUATION (Vectorized Map) ---
@jit
def evaluate_single_maze(q_table, maze):
    """Evaluates a single maze greedily. Returns 1 if goal reached, 0 otherwise, and steps taken."""
    max_steps = GRID_SIZE * GRID_SIZE * 4
    init_state = (START, 0, False, False) # (pos, steps, hit_wall, reached_goal)
    
    def cond(state):
        _, steps, hit_wall, reached_goal = state
        return (steps < max_steps) & ~hit_wall & ~reached_goal
        
    def body(state):
        pos, steps, _, _ = state
        w, dr, dc = get_state_4d(maze, pos)
        
        # Greedy tie-breaking action selection (epsilon=0)
        q_values = q_table[w, dr, dc]
        dr_goal, dc_goal = jnp.sign(GOAL[0] - pos[0]), jnp.sign(GOAL[1] - pos[1])
        bonus = jnp.array([
            (ACTION_DELTAS[0][0] == dr_goal) * 1e-5,
            (ACTION_DELTAS[1][0] == dr_goal) * 1e-5,
            (ACTION_DELTAS[2][1] == dc_goal) * 1e-5,
            (ACTION_DELTAS[3][1] == dc_goal) * 1e-5,
        ])
        action = jnp.argmax(q_values + bonus)
        
        new_pos, _, done = step(maze, pos, action)
        
        # Check termination reasons
        hit_wall = done & ~((new_pos[0] == GOAL[0]) & (new_pos[1] == GOAL[1]))
        reached_goal = (new_pos[0] == GOAL[0]) & (new_pos[1] == GOAL[1])
        
        return (new_pos, steps + 1, hit_wall, reached_goal)
        
    final_state = lax.while_loop(cond, body, init_state)
    _, final_steps, _, reached_goal = final_state
    
    # Return 1.0 for success, 0.0 for failure
    return jnp.where(reached_goal, 1.0, 0.0), final_steps

# MAGIC: `jax.vmap` turns `evaluate_single_maze` into a batch processor!
# in_axes=(None, 0) means: don't batch the Q-table, DO batch the mazes array along axis 0.
batch_evaluate = jit(vmap(evaluate_single_maze, in_axes=(None, 0)))

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load your training mazes ---
    train_maze_path = 'data_jax/N16_P0100_train_solvable.npy'
    try:
        train_mazes = np.load(train_maze_path)
        print(f"Loaded {len(train_mazes)} mazes for training.")
    except:
        print("Fallback to dummy mazes for testing the script...")
        train_mazes = np.random.choice([0, 1], size=(100, 16, 16), p=[0.8, 0.2])
        train_mazes[:, 0, 0] = 0; train_mazes[:, 15, 15] = 0

    # --- 2. Train and Save 4D JAX Array ---
    q_table_4d, rewards = train_jax(train_mazes, n_episodes=50000)
    
    # ---------------------------------------------------------
    # --- NEW: REGISTER IN ALIOS ---
    # ---------------------------------------------------------
    print("\n📦 Registering Run to Alios Database...")
    
    # Flatten (512, 3, 3, 4) to (4608, 4) for Alios compatibility
    q_table_2d = np.array(q_table_4d).reshape(-1, 4)
    
    run_config = {
        "episodes": 50000,
        "alpha": 0.2,
        "gamma": 0.99,
        "reward_shaping": "Progress +0.5, Step -0.2, Wall -10.0",
        "epsilon_decay": "1.0 to 0.01",
        "notes": "Pure JAX 4D Q-Learning, trained on P0100"
    }

    register_run(
        run_id="JAX_4D_QLearning_002",
        algo="Q-Learning",
        state_repr="pomdp_9bit_compass",
        config_dict=run_config,
        q_table=q_table_2d
    )
    # ---------------------------------------------------------
