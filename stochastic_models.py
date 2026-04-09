import jax
import jax.numpy as jnp

class StatefulPolicy:
    def init_memory(self, batch_size):
        return jnp.full((batch_size,), -1, dtype=jnp.int32)

    # UNIFIED API: Every policy now receives (maze, pos, memory, params)
    def get_action_probs(self, maze, pos, memory, params):
        raise NotImplementedError

    def step(self, rng_key, maze, pos, memory, params):
        probs = self.get_action_probs(maze, pos, memory, params)
        action = jax.random.categorical(rng_key, jnp.log(probs + 1e-9))
        return action, probs, action # memory updates to last action

class EgoPolicy(StatefulPolicy):
    def get_action_probs(self, maze, pos, memory, params):
        # 1. Extract Window internally!
        r, c = pos[0], pos[1]
        padded = jnp.pad(maze, 1, constant_values=1)
        window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
        powers = jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1], dtype=jnp.int32)
        win_id = jnp.sum(window.astype(jnp.int32) * powers)
        
        # 2. Compute Probs
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
        
        exp_q = jnp.exp(kappa * q_values)
        return exp_q / jnp.sum(exp_q)