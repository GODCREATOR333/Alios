import jax
import jax.numpy as jnp
from state_sandbox import decode_meta_state

class StatefulPolicy:
    def init_memory(self, batch_size):
        return jnp.full((batch_size,), -1, dtype=jnp.int32)

    def get_action_probs(self, maze, pos, memory, params):
        raise NotImplementedError

    def update_memory(self, rng_key, action, hit, maze, pos, memory, params):
        """Default: just remember the last action."""
        return jnp.where(hit, memory, action)

class EgoPolicy(StatefulPolicy):
    def get_action_probs(self, maze, pos, memory, params):
        r, c = pos[0], pos[1]
        padded = jnp.pad(maze, 1, constant_values=1)
        window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
        powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
        win_id = jnp.sum(window.astype(jnp.int32) * powers)
        
        D_R = params.get('D_R', 0.15)
        p_straight = jnp.exp(-D_R)
        p_turn = (1.0 - p_straight) / 2.0

        is_wall = jnp.array([(win_id >> 7) & 1, (win_id >> 1) & 1, (win_id >> 5) & 1, (win_id >> 3) & 1])

        def calc_probs(la):
            opp_idx = jnp.take(jnp.array([1, 0, 3, 2]), la)
            base = jnp.full(4, p_turn)
            base = base.at[la].set(p_straight)
            base = base.at[opp_idx].set(0.0)
            return base

        probs = jnp.where(memory == -1, jnp.full(4, 0.25), calc_probs(memory))
        probs = jnp.where(is_wall, 0.0, probs)
        s = jnp.sum(probs)
        
        rev_idx = jnp.take(jnp.array([1, 0, 3, 2]), jnp.maximum(0, memory))
        fallback = jnp.zeros(4).at[rev_idx].set(1.0)
        return jnp.where(s > 0, probs / s, fallback)

class GeoPolicy(StatefulPolicy):
    def get_action_probs(self, maze, pos, memory, params):
        kappa = params.get('kappa', 2.5)
        goal_pos = params.get('goal_pos', jnp.array([15.0, 15.0]))
        
        pos_float = pos.astype(jnp.float32)
        v_g = goal_pos - pos_float
        norm = jnp.linalg.norm(v_g)
        u_g = jnp.where(norm > 0, v_g / norm, jnp.zeros(2))
        
        actions = jnp.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])
        q_values = jnp.dot(actions, u_g)
        return jax.nn.softmax(kappa * q_values)

class HybridPolicy(StatefulPolicy):
    def init_memory(self, batch_size):
        # Memory: [Mode, Last_Action]. Mode 0 = GEO, Mode 1 = EGO.
        return jnp.tile(jnp.array([0, -1], dtype=jnp.int32), (batch_size, 1))

    def get_action_probs(self, maze, pos, memory, params):
        mode = memory[0]
        last_a = memory[1]
        
        # Calculate both
        geo_probs = GeoPolicy().get_action_probs(maze, pos, memory, params)
        ego_probs = EgoPolicy().get_action_probs(maze, pos, last_a, params)
        
        return jnp.where(mode == 0, geo_probs, ego_probs)

    def update_memory(self, rng_key, action, hit, maze, pos, memory, params):
        mode = memory[0]
        last_a = memory[1]
        gamma = params.get('gamma', 0.5)

        # If GEO (0)
        geo_next_mode = jnp.where(hit, 1, 0)
        geo_next_last_a = jnp.where(hit, action, -1)

        # If EGO (1)
        switch_to_geo = jax.random.bernoulli(rng_key, gamma)
        ego_next_mode = jnp.where(switch_to_geo, 0, 1)
        ego_next_last_a = jnp.where(hit, last_a, action)

        next_mode = jnp.where(mode == 0, geo_next_mode, ego_next_mode)
        next_last_a = jnp.where(mode == 0, geo_next_last_a, ego_next_last_a)
        
        return jnp.array([next_mode, next_last_a], dtype=jnp.int32)
    

class LearnedMetaPolicy(StatefulPolicy):
    """
    The End-Game Agent: Uses a Learned Q-Table to switch between GEO and EGO physics.
    Memory: [Current_Mode, Last_Action]
    """
    def __init__(self, q_table):
        self.q_table = q_table

    def init_memory(self, batch_size):
        # Initial: [Mode: GEO, LastAction: 4 (None)]
        return jnp.tile(jnp.array([0, 4], dtype=jnp.int32), (batch_size, 1))

    def get_action_probs(self, maze, pos, memory, params):
        r, c = pos[0], pos[1]
        last_a = memory[1]
        
        # 1. Look up the Meta-Choice (GEO vs EGO) from the Q-Table
        state_id = decode_meta_state(maze, r, c, last_a)
        q_vals = self.q_table[state_id]
        meta_action = jnp.argmax(q_vals) # 0: GEO, 1: EGO
        
        # 2. Get probabilities from both physical modes
        geo_probs = GeoPolicy().get_action_probs(maze, pos, memory, params)
        # Ego expects 0-3, we handle 4 (None) internally
        ego_probs = EgoPolicy().get_action_probs(maze, pos, last_a, params)
        
        # 3. Return the distribution of the chosen mode
        return jnp.where(meta_action == 0, geo_probs, ego_probs)

    def update_memory(self, rng_key, action, hit, maze, pos, memory, params):
        # We need to recalculate the state to find WHICH mode the Q-table chose
        # so we can store it in the memory for the next step/visualization
        r, c = pos[0], pos[1]
        last_a = memory[1]
        state_id = decode_meta_state(maze, r, c, last_a)
        
        # Determine which mode the Q-table preferred at THIS step
        mode = jnp.argmax(self.q_table[state_id]) 
        
        # Store [Mode, Physical_Action]
        return jnp.array([mode, action], dtype=jnp.int32)