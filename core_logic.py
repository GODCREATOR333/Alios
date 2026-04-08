# physics_engine.py
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit

# ==========================================================
# 1. AUTO-REGISTRY SYSTEM
# ==========================================================
DECODERS = {}
STATE_MAP_FUNCS_RAW = {}
STATE_MAP_FUNCS_JIT = {}

def register_state(name_key):
    """
    Decorator that automatically registers a custom state representation.
    It takes a single (r, c) decoder and automatically generates the 
    JAX vmap (Raw) and JIT-compiled versions for UI and Batch processing.
    """
    def decorator(scalar_func):
        # 1. Save the basic single-cell function for the UI Click-Probe
        DECODERS[name_key] = scalar_func
        
        # 2. Automatically generate the vectorized whole-grid mapper
        def grid_mapper(maze):
            r_idx, c_idx = jnp.indices((16, 16))
            return jax.vmap(jax.vmap(scalar_func, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)
            
        # 3. Save the Raw version (used for Batch Evaluation on 1000 mazes)
        STATE_MAP_FUNCS_RAW[name_key] = grid_mapper
        
        # 4. Save the compiled JIT version (used for lightning-fast UI rendering)
        STATE_MAP_FUNCS_JIT[name_key] = jax.jit(grid_mapper)
        
        return scalar_func
    return decorator


# ==========================================================
# 2. STATISTICAL EVALUATOR (The Batch World Simulator)
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
    state_map = state_map_func(maze)
    actions = jnp.argmax(q_table[state_map], axis=-1)
    
    dr = jnp.array([-1, 1, 0, 0])[actions]
    dc = jnp.array([0, 0, -1, 1])[actions]
    
    r, c = jnp.indices((16, 16))
    nr, nc = r + dr, c + dc
    
    out = (nr < 0) | (nr >= 16) | (nc < 0) | (nc >= 16)
    safe_nr, safe_nc = jnp.clip(nr, 0, 15), jnp.clip(nc, 0, 15)
    hits_wall = out | (maze[safe_nr, safe_nc] == 1)
    
    is_path = (maze == 0) & ~((r == 15) & (c == 15))
    
    unsafe_cells = jnp.sum(hits_wall & is_path)
    total_path_cells = jnp.sum(is_path)
    
    return (unsafe_cells / total_path_cells) * 100


# ==========================================================
# 3. MATHEMATICAL PROBES & UI HELPERS
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


# ==========================================================
# 4. ADVANCED PHYSICS LENSES (Localization, IPR, MSD)
# ==========================================================
@partial(jax.jit, static_argnums=(2,))
def calculate_localization_batch(q_table, dataset_jax, state_map_func):
    all_state_maps = jax.vmap(state_map_func)(dataset_jax)
    v_maps = jnp.max(q_table[all_state_maps], axis=-1)
    
    v_density = jnp.exp(v_maps / 10.0) 
    v_flat = v_density.reshape(v_density.shape[0], -1)
    
    sum_sq = jnp.sum(v_flat**2, axis=-1)
    sum_quad = jnp.sum(v_flat**4, axis=-1)
    
    pr = (sum_sq**2) / (sum_quad + 1e-9)
    return pr / 256.0

def calculate_path_correlation_batch(final_results):
    steps = np.array(final_results[2])
    return np.mean(steps)

@partial(jax.jit, static_argnums=(2,))
def compute_transition_matrix_batch(q_table, mazes, state_map_func):
    batch_size = mazes.shape[0]
    r_idx, c_idx = jnp.indices((16, 16))
    
    state_maps = jax.vmap(state_map_func)(mazes)
    actions = jnp.argmax(q_table[state_maps], axis=-1) 
    actions_flat = actions.reshape(batch_size, -1)     
    
    def get_targets(maze, acts):
        dr = jnp.array([-1, 1, 0, 0])[acts]
        dc = jnp.array([0, 0, -1, 1])[acts]
        nr, nc = r_idx.flatten() + dr, c_idx.flatten() + dc
        
        out = (nr < 0) | (nr >= 16) | (nc < 0) | (nc >= 16)
        hit = out | (maze[jnp.clip(nr, 0, 15), jnp.clip(nc, 0, 15)] == 1)
        
        target_r = jnp.where(hit, r_idx.flatten(), nr)
        target_c = jnp.where(hit, c_idx.flatten(), nc)
        return target_r * 16 + target_c

    target_states = jax.vmap(get_targets)(mazes, actions_flat) 
    goal_state_id = 15 * 16 + 15
    target_states = target_states.at[:, goal_state_id].set(goal_state_id)
    
    P = jnp.zeros((batch_size, 256, 256))
    batch_indices = jnp.arange(batch_size)[:, None]
    state_indices = jnp.arange(256)[None, :]
    P = P.at[batch_indices, state_indices, target_states].set(1.0)
    
    return P

@jax.jit
def calculate_ipr_statistics(P_matrices, steps=512):
    batch_size = P_matrices.shape[0]
    v_init = jnp.zeros((batch_size, 256))
    v_init = v_init.at[:, 0].set(1.0)
    
    window_start = 400
    num_summed_steps = steps - window_start - 1 

    def body(i, carry):
        v_curr, v_sum = carry
        v_next = jnp.einsum('bi,bij->bj', v_curr, P_matrices)
        is_steady = i > window_start 
        v_sum = jnp.where(is_steady, v_sum + v_next, v_sum)
        return (v_next, v_sum)

    _, v_total = jax.lax.fori_loop(0, steps, body, (v_init, jnp.zeros_like(v_init)))
    v_avg = v_total / num_summed_steps
    return jnp.sum(v_avg**2, axis=-1)

@jit
def calculate_msd(rollout_results):
    final_r = rollout_results[0]
    final_c = rollout_results[1]
    squared_displacement = (final_r - 0)**2 + (final_c - 0)**2
    return jnp.mean(squared_displacement)