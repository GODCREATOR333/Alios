# swarm_engine.py
import jax
import jax.numpy as jnp
from functools import partial

# =======================================================================
# 1. SENSORS (Converting Environment to Policy Inputs)
# =======================================================================

@jax.jit
def get_obs_window(maze, pos):
    """Extracts 9-bit window ID (0-511) for EGO agents."""
    r, c = pos[0], pos[1]
    padded = jnp.pad(maze, 1, constant_values=1)
    # Extract 3x3 window
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
    return jnp.sum(window.astype(jnp.int32) * powers)

@jax.jit
def get_obs_mdp(maze, pos):
    """Standard (r, c) -> 0-255 state ID."""
    return jnp.int32(pos[0] * 16 + pos[1])

# =======================================================================
# 2. THE SWARM SIMULATOR (Monte Carlo Ghost Cloud)
# =======================================================================

@partial(jax.jit, static_argnums=(2, 5, 6)) 
def simulate_swarm(rng_key, maze, policy, params, start_pos, num_ghosts=100, max_steps=150):
    keys = jax.random.split(rng_key, num_ghosts)
    """
    Runs num_ghosts in parallel from a single start point.
    Returns:
        paths: (num_ghosts, max_steps, 2) -> Every coordinate for every ghost
        occupancy: (16, 16) -> Heatmap of where they spent time
    """
    keys = jax.random.split(rng_key, num_ghosts)
    
    # We use vmap to turn a single ghost into a swarm
    # we must partial the maze and policy because they are static
    def single_ghost_rollout(key, m, p, par, start):
        init_mem = p.init_memory(1)[0]
        
        def step_fn(carry, step_idx):
            pos, mem, k = carry
            k, subkey = jax.random.split(k)
            
            # Simple sensor logic
            obs = get_obs_window(m, pos) 
            action, probs, next_mem = p.step(subkey, obs, mem, par)
            
            dr = jnp.array([-1, 1, 0, 0])[action]
            dc = jnp.array([0, 0, -1, 1])[action]
            npos = pos + jnp.array([dr, dc])
            
            hit = (npos[0]<0)|(npos[0]>=16)|(npos[1]<0)|(npos[1]>=16) | (m[npos[0], npos[1]]==1)
            final_pos = jnp.where(hit, pos, npos)
            
            return (final_pos, next_mem, k), final_pos

        _, path = jax.lax.scan(step_fn, (start, init_mem, key), jnp.arange(max_steps))
        return path

    # Use vmap correctly by specifying which arguments to map over
    # (key=mapped, maze=None, policy=None, params=None, start_pos=None)
    all_paths = jax.vmap(single_ghost_rollout, in_axes=(0, None, None, None, None))(
        keys, maze, policy, params, start_pos
    )
    
    # Calculate Occupancy Heatmap
    # We flatten all coordinates and count occurrences
    flat_paths = all_paths.reshape(-1, 2)
    # Note: Bincount in JAX is a bit tricky for 2D, so we map to 1D index
    flat_indices = flat_paths[:, 0] * 16 + flat_paths[:, 1]
    counts = jnp.bincount(flat_indices, length=256)
    occupancy = counts.reshape(16, 16)
    
    return all_paths, occupancy

# =======================================================================
# 3. FLUID DYNAMICS (Expected Vector Field)
# =======================================================================


@partial(jax.jit, static_argnums=(1,)) 
def compute_vector_field(maze, policy, params, memory_val=-1):
    """
    Calculates the 'Fluid Flow' of the agent.
    For every cell, what is the Mean Resultant Vector <v>?
    """
    r_idx, c_idx = jnp.indices((16, 16))
    
    def get_cell_v(r, c):
        pos = jnp.array([r, c])
        obs = get_obs_window(maze, pos) if hasattr(policy, 'get_obs_window') else get_obs_mdp(maze, pos)
        
        # We assume a fixed memory (e.g. agent was moving UP) to see the vector field
        probs = policy.get_action_probs(obs, memory_val, params)
        
        # Directions: UP, DOWN, LEFT, RIGHT
        v_actions = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        
        # Mean Vector = Sum(P(a) * Vector(a))
        mean_v = jnp.dot(probs, v_actions)
        return mean_v, jnp.max(probs) # Return mean vector and "Confidence"

    # Map across the whole grid
    field, confidence = jax.vmap(jax.vmap(get_cell_v))(r_idx, c_idx)
    return field, confidence