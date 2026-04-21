import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2, 5, 6)) 
def simulate_swarm(rng_key, maze, policy, params, start_pos, num_ghosts=100, max_steps=150):
    keys = jax.random.split(rng_key, num_ghosts)
    
    def single_ghost_rollout(key, m, p, par, start):
        start = start.astype(jnp.int32)
        init_mem = p.init_memory(1)[0]
        
        # carry = (position, memory, random_key, done_flag, collision_count)
        def step_fn(carry, step_idx):
            pos, mem, k, done, colls = carry
            k, sk1, sk2 = jax.random.split(k, 3)
            
            # 1. Ask Policy what to do
            probs = p.get_action_probs(m, pos, mem, par)
            action = jax.random.categorical(sk1, jnp.log(probs + 1e-9))
            
            # 2. Physics: Move
            dr = jnp.array([-1, 1, 0, 0])[action]
            dc = jnp.array([0, 0, -1, 1])[action]
            npos = pos + jnp.array([dr, dc])
            
            # 3. Collision Logic
            hit = (npos[0]<0)|(npos[0]>=16)|(npos[1]<0)|(npos[1]>=16) | (m[npos[0], npos[1]]==1)
            
            # 4. Terminal State Logic (The Fix)
            # If already done, stay where you are. Otherwise, apply collision physics.
            final_pos = jnp.where(done, pos, jnp.where(hit, pos, npos))
            
            # Check if we just reached the goal
            is_goal = (final_pos[0] == 15) & (final_pos[1] == 15)
            next_done = done | is_goal
            
            # Update Memory (Only if we aren't done)
            next_mem = jnp.where(done, mem, p.update_memory(sk2, action, hit, m, pos, mem, par))
            
            # Update Collisions
            next_colls = colls + jnp.where(hit & ~done, 1, 0)
            
            next_carry = (final_pos, next_mem, k, next_done, next_colls)
            valid_hit = hit & ~done 
            record = (final_pos, mem, action, next_done, valid_hit)
            
            return next_carry, record

        _, (paths, mems, acts, dones, hits) = jax.lax.scan(step_fn, (start, init_mem, key, False, 0), jnp.arange(max_steps))
        return paths, mems, acts, dones, hits

    # Vmap across the keys
    all_paths, all_mems, all_acts, all_dones, all_hits = jax.vmap(single_ghost_rollout, in_axes=(0, None, None, None, None))(
        keys, maze, policy, params, start_pos
    )
    
    # Heatmap logic
    flat_paths = all_paths.reshape(-1, 2)
    flat_indices = flat_paths[:, 0] * 16 + flat_paths[:, 1]
    counts = jnp.bincount(flat_indices, length=256)
    occupancy = counts.reshape(16, 16)
    
    return all_paths, all_mems, all_acts, all_dones, all_hits, occupancy

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