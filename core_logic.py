import jax.numpy as jnp
import jax

import jax.numpy as jnp
import jax

@jax.jit
def get_full_state_map_pomdp(maze):
    """Calculates all 256 state IDs for a maze in one JAX call."""
    # 1. Goal Injection & Padding
    maze_with_goal = maze.at[15, 15].set(2)
    padded = jnp.pad(maze_with_goal, 1, constant_values=1)
    
    # 2. Use JAX 'vmap' or sliding window to get all 3x3 windows at once
    # We'll use a trick with jax.lax.conv to get the local IDs efficiently
    # but for now, we'll keep it simple with a 2D window extraction
    
    # Get indices for all 256 cells
    r, c = jnp.indices((16, 16))
    
    
    def single_state(ri, ci):
        window = jax.lax.dynamic_slice(padded, (ri, ci), (3, 3))
        flat = window.flatten()
        neighbors = jnp.concatenate([flat[:4], flat[5:]])
        powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1])
        local_id = jnp.sum(neighbors * powers)
        dr = jnp.where(ri < 15, 1, 0)
        dc = jnp.where(ci < 15, 1, 0)
        compass_id = dr * 2 + dc
        return (local_id * 4 + compass_id).astype(jnp.int32)

    # Vectorize the function over the grid
    return jax.vmap(jax.vmap(single_state))(r, c)

@jax.jit
def get_full_state_map_mdp(maze):
    r, c = jnp.indices((16, 16))
    return r * 16 + c

# Factory
STATE_MAP_FUNCS = {
    "mdp": get_full_state_map_mdp,
    "3x3_base3_compass": get_full_state_map_pomdp
}


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