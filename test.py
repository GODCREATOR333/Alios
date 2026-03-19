import numpy as np

# Load data
data = np.load("data_jax/N16_P0100_test_solvable_random.npy")

print("Original shape:", data.shape)  # (1000, 16, 16)

# # Create empty maze (all free)
# empty_maze = np.zeros((16, 16), dtype=data.dtype)

# # Ensure start & goal are free (just being explicit)
# empty_maze[0, 0] = 0
# empty_maze[15, 15] = 0

# # Add new maze
# data_new = np.concatenate([data, empty_maze[np.newaxis, ...]], axis=0)

# print("New shape:", data_new.shape)  # (1001, 16, 16)

# # Save back
# np.save("data_jax/N16_P0100_test_with_empty.npy", data_new)