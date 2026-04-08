# tests/test_core.py
import jax.numpy as jnp
import numpy as np

# Import the core and the sandbox to trigger the registry
import core_logic
import state_sandbox 

def test_registry_populated():
    """Checks if the @register_state decorator successfully linked the functions."""
    assert "mdp" in core_logic.DECODERS
    assert "pure_ego" in core_logic.STATE_MAP_FUNCS_JIT
    assert len(core_logic.DECODERS) >= 6  # Ensure all default ones loaded

def test_jax_mapper_compilation():
    """Tests if the JAX JIT compiled mappers actually work on a dummy maze."""
    # Create an empty 16x16 maze
    dummy_maze = jnp.zeros((16, 16), dtype=jnp.int32)
    
    # Grab the JIT-compiled MDP mapper
    mapper = core_logic.STATE_MAP_FUNCS_JIT["mdp"]
    
    # Run it!
    state_map = mapper(dummy_maze)
    
    # Assertions: Did it output the correct shape?
    assert state_map.shape == (16, 16), "State map should match maze dimensions"
    
    # In MDP, cell (0, 0) is state 0, cell (15, 15) is state 255
    assert state_map[0, 0] == 0
    assert state_map[15, 15] == 255

def test_entropy_math():
    """Tests if the mathematical probes are computing correctly."""
    # A Q-table vector where all actions have the exact same value
    q_vals = np.array([1.0, 1.0, 1.0, 1.0])
    
    entropy = core_logic.calculate_entropy(q_vals)
    
    # Max entropy for 4 choices is roughly 1.386 (which is ln(4))
    assert np.isclose(entropy, 1.386, atol=0.01), f"Expected ~1.386, got {entropy}"