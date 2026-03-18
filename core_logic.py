import jax.numpy as jnp
import jax

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