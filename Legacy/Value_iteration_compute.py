import jax
import jax.numpy as jnp
import numpy as np
import os

def solve_value_iteration(dataset_path, output_path):
    # 1. Load Dataset
    data = np.load(dataset_path)
    # Ensure it's 3D (Batch, R, C)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    data_jax = jnp.array(data)
    num_mazes = data_jax.shape[0]
    
    # 2. Hyperparameters
    gamma = 1.0
    reward = -1.0
    # 500 iterations is the 'Safe Physics' limit for 16x16 to ensure 0% solid colors
    num_iterations = 500 

    # 3. Initialization
    # Initialize all to 0, Walls to -inf, Goal to 0
    V = jnp.zeros_like(data_jax, dtype=jnp.float32)
    V = jnp.where(data_jax == 1, -jnp.inf, V)
    V = V.at[:, 15, 15].set(0.0)

    # 4. The JAX Vectorized Update Loop
    def body_fun(i, V_curr):
        # Shift matrices to get neighbors
        V_up    = jnp.roll(V_curr, shift=1, axis=1)
        V_down  = jnp.roll(V_curr, shift=-1, axis=1)
        V_left  = jnp.roll(V_curr, shift=1, axis=2)
        V_right = jnp.roll(V_curr, shift=-1, axis=2)

        # Mask edges (Corrected logic to prevent wrap-around signal)
        V_up    = V_up.at[:, 0, :].set(-jnp.inf)
        V_down  = V_down.at[:, -1, :].set(-jnp.inf)
        V_left  = V_left.at[:, :, 0].set(-jnp.inf)
        V_right = V_right.at[:, :, -1].set(-jnp.inf)

        # Bellman Optimality Equation: V = R + max(V_neighbors)
        V_stack = jnp.stack([V_up, V_down, V_left, V_right], axis=-1)
        best_next_V = jnp.max(V_stack, axis=-1)
        
        new_V = reward + gamma * best_next_V

        # Re-enforce Guardrails: Walls stay -inf, Goal stays 0
        new_V = jnp.where(data_jax == 1, -jnp.inf, new_V)
        new_V = new_V.at[:, 15, 15].set(0.0)
        
        return new_V

    print(f"⏳ Solving {num_mazes} mazes via JAX...")
    # Use JIT and fori_loop for M4-speed
    V_final = jax.jit(lambda v_init: jax.lax.fori_loop(0, num_iterations, body_fun, v_init))(V)

    # 5. Extract Policy (Argmax over the final V)
    V_up    = jnp.roll(V_final, shift=1, axis=1).at[:, 0, :].set(-jnp.inf)
    V_down  = jnp.roll(V_final, shift=-1, axis=1).at[:, -1, :].set(-jnp.inf)
    V_left  = jnp.roll(V_final, shift=1, axis=2).at[:, :, 0].set(-jnp.inf)
    V_right = jnp.roll(V_final, shift=-1, axis=2).at[:, :, -1].set(-jnp.inf)
    
    policy = jnp.argmax(jnp.stack([V_up, V_down, V_left, V_right], axis=-1), axis=-1)

    # 6. Save as .npz for ALIOS
    np.savez(output_path, values=np.array(V_final), policy=np.array(policy))
    print(f"✅ Success! Saved Oracle to: {output_path}")

if __name__ == "__main__":
    # Example: Regenerate the empty maze first to test
    solve_value_iteration(
        "data_jax/N16_P0000_empty_test.npy", 
        "data_jax/value_iteration/N16_P0000_empty_test_VI_solved.npz"
    )