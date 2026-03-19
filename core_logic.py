import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# ==========================================================
# 1. STATE DECODERS (JAX-Pure)
# ==========================================================

def decode_mdp(maze, r, c):
    """Classic Grid MDP: 0-255"""
    # Use .astype instead of int()
    return (r * 16 + c).astype(jnp.int32)

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

def decode_legacy_9bit(maze, r, c):
    """Legacy: 9-bit window (0/1) + signed compass"""
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    win_id = jnp.sum(window.astype(jnp.int32) * powers)
    
    # sign() returns float tracers, so we keep them as tracers for the math
    dr = jnp.sign(15 - r)
    dc = jnp.sign(15 - c)
    comp_id = (dr + 1) * 3 + (dc + 1)
    
    return (win_id * 9 + comp_id).astype(jnp.int32)

# ==========================================================
# 2. VECTORIZED STATE MAPPERS (JIT)
# ==========================================================

@jax.jit
def get_full_state_map_mdp(maze):
    r, c = jnp.indices((16, 16))
    return decode_mdp(maze, r, c)

@jax.jit
def get_full_state_map_pomdp(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_pomdp_3x3_base3, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)

@jax.jit
def get_full_state_map_legacy(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_legacy_9bit, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)

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

# ==========================================================
# 5. REGISTRIES
# ==========================================================

DECODERS = {
    "mdp": decode_mdp,
    "3x3_base3_compass": decode_pomdp_3x3_base3,
    "legacy_9bit_compass": decode_legacy_9bit,
}

STATE_MAP_FUNCS_RAW = {
    "mdp": get_full_state_map_mdp,
    "3x3_base3_compass": get_full_state_map_pomdp,
    "legacy_9bit_compass": get_full_state_map_legacy
}

STATE_MAP_FUNCS_JIT = {
    "mdp": get_full_state_map_mdp, # Already jitted above
    "3x3_base3_compass": get_full_state_map_pomdp,
    "legacy_9bit_compass": get_full_state_map_legacy
}