import jax.numpy as jnp
import jax
from functools import partial
import pickle
import os
import numpy as np


def get_full_state_map_pomdp(maze):
    maze_with_goal = maze.at[15, 15].set(2)
    padded = jnp.pad(maze_with_goal, 1, constant_values=1)
    r_indices, c_indices = jnp.indices((16, 16))
    
    def get_single_state(ri, ci):
        window = jax.lax.dynamic_slice(padded, (ri, ci), (3, 3))
        flat = window.flatten()
        neighbors = jnp.concatenate([flat[:4], flat[5:]])
        powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1])
        local_id = jnp.sum(neighbors * powers)
        dr = jnp.where(ri < 15, 1, 0)
        dc = jnp.where(ci < 15, 1, 0)
        compass_id = dr * 2 + dc
        return (local_id * 4 + compass_id).astype(jnp.int32)

    return jax.vmap(jax.vmap(get_single_state))(r_indices, c_indices)


def get_full_state_map_mdp(maze):
    r, c = jnp.indices((16, 16))
    return (r * 16 + c).astype(jnp.int32)


def decode_mdp(maze, r, c):
    return int(r * 16 + c)


def decode_pomdp_3x3_base3(maze, r, c):
    maze_with_goal = jnp.array(maze).at[15, 15].set(2)
    padded = jnp.pad(maze_with_goal, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3))
    flat = window.flatten()
    neighbors = jnp.concatenate([flat[:4], flat[5:]])
    powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1])
    local_id = jnp.sum(neighbors * powers)
    dr = 1 if r < 15 else 0
    dc = 1 if c < 15 else 0
    compass_id = dr * 2 + dc
    return int(local_id * 4 + compass_id)


@partial(jax.jit, static_argnums=(2,))
def evaluate_dataset(q_table, dataset_jax, state_map_func):
    all_state_maps = jax.vmap(state_map_func)(dataset_jax)
    all_actions = jnp.argmax(q_table[all_state_maps], axis=-1)

    def single_maze_rollout(maze, actions):
        def cond(state): 
            return (state[2] < 500) & (~state[4])

        def body(state):
            r, c, steps, collisions, done = state
            action = actions[r, c]
            
            dr = jnp.array([-1, 1, 0, 0])[action]
            dc = jnp.array([0, 0, -1, 1])[action]
            nr, nc = r + dr, c + dc
            
            hit_wall = (nr<0)|(nr>=16)|(nc<0)|(nc>=16) | (maze[jnp.clip(nr,0,15), jnp.clip(nc,0,15)]==1)
            
            fr, fc = jnp.where(hit_wall, r, nr), jnp.where(hit_wall, c, nc)
            is_goal = (fr==15)&(fc==15)
            
            return (fr, fc, steps + 1, collisions + jnp.where(hit_wall, 1, 0), is_goal)

        init = (0, 0, 0, 0, False)
        return jax.lax.while_loop(cond, body, init)

    results = jax.vmap(single_maze_rollout)(dataset_jax, all_actions)
    return results


def calculate_entropy(q_vals, temperature=1.0):
    exp_q = np.exp((q_vals - np.max(q_vals)) / temperature)
    probs = exp_q / np.sum(exp_q)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return float(entropy)

def calculate_entropy_grid(q_values_grid, temperature=1.0):
    """
    Calculates Shannon Entropy for every cell in a 16x16 grid.
    q_values_grid: (16, 16, 4)
    Returns: (16, 16) array of entropy values.
    """
    # 1. Softmax to get probabilities P(a|s)
    # Subtract max for numerical stability
    max_q = np.max(q_values_grid, axis=-1, keepdims=True)
    exp_q = np.exp((q_values_grid - max_q) / temperature)
    probs = exp_q / np.sum(exp_q, axis=-1, keepdims=True)
    
    # 2. Shannon Entropy: H = -sum(p * log(p))
    # We add a tiny epsilon to avoid log(0)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
    return entropy

def calculate_conflict(state_id, state_map, oracle_policy):
    coords = np.argwhere(state_map == state_id)
    if len(coords) <= 1:
        return 0.0 
    
    oracle_actions = [oracle_policy[r, c] for (r, c) in coords]
    
    most_common_count = max([oracle_actions.count(a) for a in set(oracle_actions)])
    agreement_rate = most_common_count / len(oracle_actions)
    
    return (1.0 - agreement_rate) * 100


def compute_rollout(maze, start_pos, q_table, decoder_func, max_steps=512):
    path = [start_pos]
    r, c = start_pos
    
    for _ in range(max_steps):
        state_id = decoder_func(maze, r, c)
        action = np.argmax(q_table[state_id])
        
        dr = [-1, 1, 0, 0][action]
        dc = [0, 0, -1, 1][action]
        nr, nc = r + dr, c + dc
        
        if nr < 0 or nr >= 16 or nc < 0 or nc >= 16 or maze[nr, nc] == 1:
            break
            
        r, c = nr, nc
        path.append((r, c))
        
        if (r, c) == (15, 15):
            break
            
    return path


# -------- LEGACY -------- #

def decode_legacy_9bit(maze, r, c):
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1])
    win_id = jnp.sum(window * powers)
    
    dr = jnp.sign(15 - r)
    dc = jnp.sign(15 - c)
    comp_id = (dr + 1) * 3 + (dc + 1)
    
    return (win_id * 9 + comp_id).astype(jnp.int32)


@jax.jit
def get_full_state_map_legacy(maze):
    r_idx, c_idx = jnp.indices((16, 16))
    return jax.vmap(jax.vmap(decode_legacy_9bit, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(maze, r_idx, c_idx)


# -------- FACTORIES -------- #

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
    "mdp": jax.jit(get_full_state_map_mdp),
    "3x3_base3_compass": jax.jit(get_full_state_map_pomdp),
    "legacy_9bit_compass": get_full_state_map_legacy
}