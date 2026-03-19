import jax.numpy as jnp
import jax
from functools import partial

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
    """Classic MDP: returns state index 0-255"""
    return int(r * 16 + c)

def decode_pomdp_3x3_base3(maze, r, c):
    """The 3x3 Window + Compass logic we developed"""
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

# The Factory: Alios looks here to find the right logic
DECODERS = {
    "mdp": decode_mdp,
    "3x3_base3_compass": decode_pomdp_3x3_base3
}


@partial(jax.jit, static_argnums=(2,))
def evaluate_dataset(q_table, dataset_jax, state_map_func):
    all_state_maps = jax.vmap(state_map_func)(dataset_jax)
    all_actions = jnp.argmax(q_table[all_state_maps], axis=-1)

    def single_maze_rollout(maze, actions):
        def cond(state): 
            # (r, c, steps, collisions, done)
            return (state[2] < 500) & (~state[4])

        def body(state):
            r, c, steps, collisions, done = state
            action = actions[r, c]
            
            dr = jnp.array([-1, 1, 0, 0])[action]
            dc = jnp.array([0, 0, -1, 1])[action]
            nr, nc = r + dr, c + dc
            
            # Physics
            hit_wall = (nr<0)|(nr>=16)|(nc<0)|(nc>=16) | (maze[jnp.clip(nr,0,15), jnp.clip(nc,0,15)]==1)
            
            fr, fc = jnp.where(hit_wall, r, nr), jnp.where(hit_wall, c, nc)
            is_goal = (fr==15)&(fc==15)
            
            # Increment collisions if hit_wall is true
            return (fr, fc, steps + 1, collisions + jnp.where(hit_wall, 1, 0), is_goal)

        # Start state: (r, c, steps, collisions, done)
        init = (0, 0, 0, 0, False)
        return jax.lax.while_loop(cond, body, init)

    results = jax.vmap(single_maze_rollout)(dataset_jax, all_actions)
    # Returns: (final_r, final_c, total_steps, total_collisions, reached_goal)
    return results




# --- FACTORIES ---

# This factory is for the Evaluator (raw functions)
STATE_MAP_FUNCS_RAW = {
    "mdp": get_full_state_map_mdp,
    "3x3_base3_compass": get_full_state_map_pomdp
}

# This factory is for the UI Slider (Jitted for speed)
STATE_MAP_FUNCS_JIT = {
    "mdp": jax.jit(get_full_state_map_mdp),
    "3x3_base3_compass": jax.jit(get_full_state_map_pomdp)
}