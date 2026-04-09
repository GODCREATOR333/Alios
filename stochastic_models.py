# stochastic_models.py
import jax
import jax.numpy as jnp

# =======================================================================
# 1. THE BASE POLICY API
# All future models (LSTMs, Belief-MDPs, Active Matter) will follow this.
# =======================================================================

class StatefulPolicy:
    """Base class for any stochastic, memory-bearing agent."""
    def init_memory(self, batch_size):
        """Returns the initial internal state (e.g., last_a = -1, or LSTM zeros)."""
        return jnp.full((batch_size,), -1, dtype=jnp.int32)

    def get_action_probs(self, obs_window, memory, params):
        """Returns the (Batch, 4) probability distribution over actions."""
        raise NotImplementedError

    def step(self, rng_key, obs_window, memory, params):
        """Samples an action and updates internal memory."""
        probs = self.get_action_probs(obs_window, memory, params)
        # Sample from categorical distribution
        action = jax.random.categorical(rng_key, jnp.log(probs + 1e-9))
        return action, probs, self.update_memory(action, obs_window, memory)

    def update_memory(self, action, obs_window, memory):
        """How the internal state updates (Default: memory = last_action)."""
        return action


# =======================================================================
# 2. THE EGO AGENT (Active Matter / Persistent Random Walk)
# =======================================================================

class EgoPolicy(StatefulPolicy):
    def get_action_probs(self, win_id, last_a, params):
        """
        params: dict containing {'D_R': float}
        """
        D_R = params.get('D_R', 0.15)
        p_straight = jnp.exp(-D_R)
        p_turn = (1.0 - p_straight) / 2.0

        # Bit decoding (UP, DOWN, LEFT, RIGHT)
        is_wall = jnp.array([
            (win_id >> 7) & 1,
            (win_id >> 1) & 1,
            (win_id >> 5) & 1,
            (win_id >> 3) & 1
        ])

        # Base probabilities
        def calc_probs(la):
            opp_idx = jnp.take(jnp.array([1, 0, 3, 2]), la) 
            base = jnp.full(4, p_turn)
            base = base.at[la].set(p_straight)
            base = base.at[opp_idx].set(0.0)
            return base

        # If last_a == -1, uniform. Else, directional inertia.
        probs = jnp.where(last_a == -1, jnp.full(4, 0.25), calc_probs(last_a))
        
        # Mask walls
        probs = jnp.where(is_wall, 0.0, probs)
        
        # Normalize (with fallback to reversing if dead-end)
        s = jnp.sum(probs)
        fallback = jnp.zeros(4).at[jnp.array([1, 0, 3, 2])[last_a]].set(1.0)
        
        return jnp.where(s > 0, probs / s, fallback)


# =======================================================================
# 3. THE GEO AGENT (Fluid Flow / Goal-Directed Softmax)
# =======================================================================

class GeoPolicy(StatefulPolicy):
    # GEO has no memory, so we just use a dummy -1 memory.
    
    def get_action_probs(self, pos, memory, params):
        """
        params: dict containing {'kappa': float, 'goal_pos': jnp.array([15, 15])}
        obs_window for GEO is just its absolute (r, c) position.
        """
        kappa = params.get('kappa', 2.5)
        goal_pos = params.get('goal_pos', jnp.array([15.0, 15.0]))
        
        pos_float = pos.astype(jnp.float32)
        v_g = goal_pos - pos_float
        norm = jnp.linalg.norm(v_g)
        
        # Avoid division by zero if at goal
        u_g = jnp.where(norm > 0, v_g / norm, jnp.zeros(2))
        
        actions = jnp.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])
        q_values = jnp.dot(actions, u_g)
        
        # Softmax / Von Mises
        exp_q = jnp.exp(kappa * q_values)
        return exp_q / jnp.sum(exp_q)


# =======================================================================
# 4. BRIDGING STATIC TO STOCHASTIC (Softmax Q-Table)
# =======================================================================

class SoftmaxQPolicy(StatefulPolicy):
    """Takes any existing .npy Q-table and turns it into a fluid distribution."""
    
    def __init__(self, q_table):
        self.q_table = q_table

    def get_action_probs(self, state_id, memory, params):
        """
        params: dict containing {'temperature': float}
        High temp = Fluid/Random. Low temp = Rigid/Greedy.
        """
        temp = params.get('temperature', 1.0)
        q_vals = self.q_table[state_id]
        
        exp_q = jnp.exp((q_vals - jnp.max(q_vals)) / temp)
        return exp_q / jnp.sum(exp_q)