# state_sandbox.py
import jax
import jax.numpy as jnp

# Import our magic decorator from the engine
from core_logic import register_state

@register_state("mdp")
def decode_mdp(maze, r, c):
    """Classic Grid MDP: 0-255"""
    return jnp.int32(r * 16 + c)

@register_state("3x3_base3_compass")
def decode_pomdp_3x3_base3(maze, r, c):
    """POMDP: 3x3 Window + Compass (Base-3 encoding)"""
    maze_with_goal = jnp.array(maze).at[15, 15].set(2)
    padded = jnp.pad(maze_with_goal, 1, constant_values=1)
    
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3))
    flat = window.flatten()
    neighbors = jnp.concatenate([flat[:4], flat[5:]])
    
    powers = jnp.array([2187, 729, 243, 81, 27, 9, 3, 1], dtype=jnp.int32)
    local_id = jnp.sum(neighbors.astype(jnp.int32) * powers)
    
    dr = jnp.where(r < 15, 1, 0)
    dc = jnp.where(c < 15, 1, 0)
    
    return (local_id * 4 + (dr * 2 + dc)).astype(jnp.int32)

@register_state("pomdp_9bit_compass")
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

@register_state("pure_ego")
def decode_pure_egocentric_3x3(maze, r, c):
    """Pure Egocentric: 9-bit window (including center). Total states: 512."""
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    return jnp.sum(window.astype(jnp.int32) * powers).astype(jnp.int32)

@register_state("pure_geo")
def decode_pure_geocentric_compass(maze, r, c):
    """Pure Geocentric: 9-way signed compass only. Total states: 9."""
    dr = jnp.sign(15 - r)
    dc = jnp.sign(15 - c)
    return ((dr + 1) * 3 + (dc + 1)).astype(jnp.int32)

@register_state("ego_persistence")
def decode_ego_persistence(maze, r, c, last_action=-1):
    """
    Heading-aware Egocentric: 3x3 window + last_action.
    Note: We map this by defaulting last_action to -1 for UI visualization.
    """
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    win_id = jnp.sum(window.astype(jnp.int32) * powers).astype(jnp.int32)
    
    memory_id = jnp.int32(last_action + 1)
    return (win_id * 5 + memory_id).astype(jnp.int32)