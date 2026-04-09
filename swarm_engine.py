import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2, 5, 6)) 
def simulate_swarm(rng_key, maze, policy, params, start_pos, num_ghosts=100, max_steps=150):
    keys = jax.random.split(rng_key, num_ghosts)
    
    def single_ghost_rollout(key, m, p, par, start):
        init_mem = p.init_memory(1)[0]
        
        def step_fn(carry, step_idx):
            pos, mem, k = carry
            k, subkey = jax.random.split(k)
            
            # UNIFIED: Just pass maze and pos. The policy handles the rest!
            action, probs, next_mem = p.step(subkey, m, pos, mem, par)
            
            dr = jnp.array([-1, 1, 0, 0])[action]
            dc = jnp.array([0, 0, -1, 1])[action]
            npos = pos + jnp.array([dr, dc])
            
            hit = (npos[0]<0)|(npos[0]>=16)|(npos[1]<0)|(npos[1]>=16) | (m[npos[0], npos[1]]==1)
            final_pos = jnp.where(hit, pos, npos)
            
            return (final_pos, next_mem, k), final_pos

        _, path = jax.lax.scan(step_fn, (start, init_mem, key), jnp.arange(max_steps))
        return path

    all_paths = jax.vmap(single_ghost_rollout, in_axes=(0, None, None, None, None))(
        keys, maze, policy, params, start_pos
    )
    
    flat_paths = all_paths.reshape(-1, 2)
    flat_indices = flat_paths[:, 0] * 16 + flat_paths[:, 1]
    counts = jnp.bincount(flat_indices, length=256)
    occupancy = counts.reshape(16, 16)
    
    return all_paths, occupancy

@partial(jax.jit, static_argnums=(1,)) 
def compute_vector_field(maze, policy, params, memory_val=-1):
    r_idx, c_idx = jnp.indices((16, 16))
    
    def get_cell_v(r, c):
        pos = jnp.array([r, c])
        probs = policy.get_action_probs(maze, pos, memory_val, params)
        v_actions = jnp.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        mean_v = jnp.dot(probs, v_actions)
        return mean_v, jnp.max(probs) 

    field, confidence = jax.vmap(jax.vmap(get_cell_v))(r_idx, c_idx)
    return field, confidence