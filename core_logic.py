import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit

# ==========================================================
# 1. STATE DECODERS (JAX-Pure)
# ==========================================================

def decode_mdp(maze, r, c):
    """Classic Grid MDP: 0-255"""
    # Change (r * 16 + c).astype(jnp.int32) to this:
    return jnp.int32(r * 16 + c)

def decode_pomdp_3x3_base3(maze, r, c):
    """POMDP: 3x3 Window + Compass (Base-3 encoding)"""
    maze_with_goal = jnp.array(maze).at[15, 15].set(2)
    padded = jnp.pad(maze_with_goal, 1, constant_values=1)
    
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3))
    flat = window.flatten()
    neighbors = jnp.concatenate([flat[:4], flat[5:]])
    
    powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1], dtype=jnp.int32)
    local_id = jnp.sum(neighbors.astype(jnp.int32) * powers)
    
    # Use jnp.where instead of Python 'if'
    dr = jnp.where(r < 15, 1, 0)
    dc = jnp.where(c < 15, 1, 0)
    
    return (local_id * 4 + (dr * 2 + dc)).astype(jnp.int32)

def decode_pomdp_9bit_compass(maze, r, c):
    """JAX-Pure: 9-bit window (0/1) + signed compass"""
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    win_id = jnp.sum(window.astype(jnp.int32) * powers)
    
    dr = jnp.sign(15 - r)
    dc = jnp.sign(15 - c)
    comp_id = (dr + 1) * 3 + (dc + 1)
    
    return (win_id * 9 + comp_id).astype(jnp.int32)

def decode_pure_egocentric_3x3(maze, r, c):
    """Pure Egocentric: 9-bit window (including center). Total states: 512."""
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    return jnp.sum(window.astype(jnp.int32) * powers).astype(jnp.int32)

def decode_pure_geocentric_compass(maze, r, c):
    """Pure Geocentric: 9-way signed compass only. Total states: 9."""
    dr = jnp.sign(15 - r)
    dc = jnp.sign(15 - c)
    return ((dr + 1) * 3 + (dc + 1)).astype(jnp.int32)


# ==========================================================
# 2. VECTORIZED STATE MAPPERS (JIT)
# ==========================================================


def get_full_state_map_mdp(maze):
    r, c = jnp.indices((16, 16))
    return decode_mdp(maze, r, c)

def get_full_state_map_pomdp(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_pomdp_3x3_base3, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)

def get_full_state_map_pomdp_9bit_compass(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_pomdp_9bit_compass, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)

def get_full_state_map_pure_ego(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_pure_egocentric_3x3, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)

def get_full_state_map_pure_geo(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_pure_geocentric_compass, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)
# ==========================================================
# 3. STATISTICAL EVALUATOR
# ==========================================================

@partial(jax.jit, static_argnums=(2,))
def evaluate_dataset(q_table, dataset_jax, state_map_func):
    all_state_maps = jax.vmap(state_map_func)(dataset_jax)
    all_actions = jnp.argmax(q_table[all_state_maps], axis=-1)

    def rollout(maze, actions):
        def cond(s): return (s[2] < 500) & (~s[4])
        def body(s):
            r, c, steps, collisions, done = s
            a = actions[r, c]
            dr, dc = jnp.array([-1, 1, 0, 0])[a], jnp.array([0, 0, -1, 1])[a]
            nr, nc = r + dr, c + dc
            hit = (nr<0)|(nr>=16)|(nc<0)|(nc>=16) | (maze[jnp.clip(nr,0,15), jnp.clip(nc,0,15)]==1)
            fr, fc = jnp.where(hit, r, nr), jnp.where(hit, c, nc)
            return (fr, fc, steps + 1, collisions + jnp.where(hit, 1, 0), (fr==15)&(fc==15))
        return jax.lax.while_loop(cond, body, (0, 0, 0, 0, False))

    return jax.vmap(rollout)(dataset_jax, all_actions)

def calculate_policy_safety(q_table, maze, state_map_func):
    """
    Checks EVERY open cell in the maze. 
    Returns the % of cells where the greedy action hits a wall.
    """
    # 1. Get the full action map for the grid
    state_map = state_map_func(maze)
    actions = jnp.argmax(q_table[state_map], axis=-1)
    
    # 2. Map actions to deltas
    dr = jnp.array([-1, 1, 0, 0])[actions]
    dc = jnp.array([0, 0, -1, 1])[actions]
    
    r, c = jnp.indices((16, 16))
    nr, nc = r + dr, c + dc
    
    # 3. Check if the destination is a wall
    out = (nr < 0) | (nr >= 16) | (nc < 0) | (nc >= 16)
    safe_nr, safe_nc = jnp.clip(nr, 0, 15), jnp.clip(nc, 0, 15)
    hits_wall = out | (maze[safe_nr, safe_nc] == 1)
    
    # 4. Only count open cells (maze == 0) and ignore the Goal (15,15)
    is_path = (maze == 0) & ~((r == 15) & (c == 15))
    
    unsafe_cells = jnp.sum(hits_wall & is_path)
    total_path_cells = jnp.sum(is_path)
    
    return (unsafe_cells / total_path_cells) * 100

# ==========================================================
# 4. MATHEMATICAL PROBES
# ==========================================================

def calculate_entropy(q_vals, temperature=1.0):
    exp_q = np.exp((q_vals - np.max(q_vals)) / temperature)
    probs = exp_q / np.sum(exp_q)
    return float(-np.sum(probs * np.log(probs + 1e-9)))

def calculate_entropy_grid(q_values_grid, temperature=1.0):
    max_q = np.max(q_values_grid, axis=-1, keepdims=True)
    exp_q = np.exp((q_values_grid - max_q) / temperature)
    probs = exp_q / np.sum(exp_q, axis=-1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-9), axis=-1)

def calculate_conflict(state_id, state_map, oracle_policy):
    coords = np.argwhere(state_map == state_id)
    if len(coords) <= 1: return 0.0 
    oracle_actions = [int(oracle_policy[r, c]) for (r, c) in coords]
    most_common_count = max([oracle_actions.count(a) for a in set(oracle_actions)])
    return (1.0 - (most_common_count / len(oracle_actions))) * 100

def compute_rollout(maze, start_pos, q_table, decoder_func, max_steps=512):
    path, (r, c) = [start_pos], start_pos
    for _ in range(max_steps):
        state_id = decoder_func(maze, r, c)
        action = int(np.argmax(q_table[state_id]))
        nr, nc = r + [-1, 1, 0, 0][action], c + [0, 0, -1, 1][action]
        if nr < 0 or nr >= 16 or nc < 0 or nc >= 16 or maze[nr, nc] == 1: break
        r, c = nr, nc
        path.append((r, c))
        if (r, c) == (15, 15): break
    return path

@partial(jax.jit, static_argnums=(2,))
def calculate_localization_batch(q_table, dataset_jax, state_map_func):
    """
    Computes the Participation Ratio (PR) for 1,000 mazes.
    PR measures how much of the environment the agent 'occupies' in its mind.
    """
    # ... (rest of the function code stays exactly the same) ...
    all_state_maps = jax.vmap(state_map_func)(dataset_jax)
    v_maps = jnp.max(q_table[all_state_maps], axis=-1)
    
    v_density = jnp.exp(v_maps / 10.0) 
    v_flat = v_density.reshape(v_density.shape[0], -1)
    
    sum_sq = jnp.sum(v_flat**2, axis=-1)
    sum_quad = jnp.sum(v_flat**4, axis=-1)
    
    pr = (sum_sq**2) / (sum_quad + 1e-9)
    return pr / 256.0

def calculate_path_correlation_batch(final_results):
    """
    Analyzes trajectories to find the 'Persistence Length'.
    Returns the average steps before directional decorrelation.
    """
    # final_results is (r_arr, c_arr, steps_arr, colls_arr, goal_arr)
    # We need the actual paths to do this perfectly, 
    # but we can estimate it using (Oracle_Steps / Actual_Steps).
    # For now, let's use the Efficiency Ratio as a proxy for 'Persistence'
    # since we aren't storing the 1000 full path buffers in RAM yet.
    
    steps = np.array(final_results[2])
    # Placeholder for a more complex auto-correlation if needed later
    return np.mean(steps)

# ==========================================================
# 5. REGISTRIES
# ==========================================================

DECODERS = {
    "mdp": decode_mdp,
    "3x3_base3_compass": decode_pomdp_3x3_base3,
    "pomdp_9bit_compass": decode_pomdp_9bit_compass,
    "pure_ego": decode_pure_egocentric_3x3,      
    "pure_geo": decode_pure_geocentric_compass 
}

# The RAW dictionary stores the pure, uncompiled Python functions for the Batch Evaluator
STATE_MAP_FUNCS_RAW = {
    "mdp": get_full_state_map_mdp,
    "3x3_base3_compass": get_full_state_map_pomdp,
    "pomdp_9bit_compass": get_full_state_map_pomdp_9bit_compass,
    "pure_ego": get_full_state_map_pure_ego,   
    "pure_geo": get_full_state_map_pure_geo,
}

# The JIT dictionary compiles them on-the-fly for the lightning-fast UI Slider
STATE_MAP_FUNCS_JIT = {
    "mdp": jax.jit(get_full_state_map_mdp), 
    "3x3_base3_compass": jax.jit(get_full_state_map_pomdp),
    "pomdp_9bit_compass": jax.jit(get_full_state_map_pomdp_9bit_compass),
    "pure_ego": jax.jit(get_full_state_map_pure_ego), 
    "pure_geo": jax.jit(get_full_state_map_pure_geo)
}




###############################################################################

# --- 1. Transition Matrix Construction ---
@partial(jax.jit, static_argnums=(2,))
def compute_transition_matrix_batch(q_table, mazes, state_map_func):
    """
    Constructs a 256x256 transition matrix P for each maze in the batch.
    Shape: (Batch, 256, 256)
    """
    batch_size = mazes.shape[0]
    r_idx, c_idx = jnp.indices((16, 16))
    source_states = (r_idx * 16 + c_idx).flatten() # 0 to 255
    
    # Get the action map for the grid
    state_maps = jax.vmap(state_map_func)(mazes)
    actions = jnp.argmax(q_table[state_maps], axis=-1) # (Batch, 16, 16)
    actions_flat = actions.reshape(batch_size, -1)     # (Batch, 256)
    
    # Calculate target states for every cell
    def get_targets(maze, acts):
        dr = jnp.array([-1, 1, 0, 0])[acts]
        dc = jnp.array([0, 0, -1, 1])[acts]
        nr, nc = r_idx.flatten() + dr, c_idx.flatten() + dc
        
        # Physics: stay if hit wall
        out = (nr < 0) | (nr >= 16) | (nc < 0) | (nc >= 16)
        hit = out | (maze[jnp.clip(nr, 0, 15), jnp.clip(nc, 0, 15)] == 1)
        
        target_r = jnp.where(hit, r_idx.flatten(), nr)
        target_c = jnp.where(hit, c_idx.flatten(), nc)
        return target_r * 16 + target_c

    target_states = jax.vmap(get_targets)(mazes, actions_flat) # (Batch, 256)
    
    # Create the sparse-style transition matrix P
    # P[batch, i, target[i]] = 1
    P = jnp.zeros((batch_size, 256, 256))
    
    # Advanced indexing to fill the matrix
    batch_indices = jnp.arange(batch_size)[:, None]
    state_indices = jnp.arange(256)[None, :]
    P = P.at[batch_indices, state_indices, target_states].set(1.0)
    
    return P

# --- 2. Inverse Participation Ratio (IPR) ---
@jit
def calculate_ipr_statistics(P_matrices, steps=512):
    """
    Evolves an initial probability distribution and calculates IPR.
    """
    batch_size = P_matrices.shape[0]
    # Start all agents at (0,0) -> Index 0
    v = jnp.zeros((batch_size, 256))
    v = v.at[:, 0].set(1.0)
    
    # Power method: v_next = v @ P
    def body(i, val):
        return jnp.einsum('bi,bij->bj', val, P_matrices)
    
    # Evolve for 512 steps to reach steady state or loop
    v_final = jax.lax.fori_loop(0, steps, body, v)
    
    # IPR = Sum of squared probabilities
    ipr = jnp.sum(v_final**2, axis=-1)
    return ipr

# --- 3. Mean Squared Displacement (MSD) ---
@jit
def calculate_msd(rollout_results):
    """
    Computes MSD from JAX evaluate_dataset results.
    rollout_results[0] and [1] are final_r and final_c.
    """
    final_r = rollout_results[0]
    final_c = rollout_results[1]
    # Displacement from (0,0)
    squared_displacement = (final_r - 0)**2 + (final_c - 0)**2
    return jnp.mean(squared_displacement)