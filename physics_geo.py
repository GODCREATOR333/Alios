import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Physics Parameters
# =========================
KAPPA = 2.5
GOAL_POS = np.array([15, 15])
MAX_STEPS = 400
DATA_PATH = "data_jax/N16_P0300_test_solvable_random.npy"

# =========================
# 2. Analytical Stochastic Policy (Blind to Obstacles)
# =========================
def get_geo_probabilities(pos, goal_pos=GOAL_POS, kappa=KAPPA):
    """
    Computes the discrete Von Mises / softmax-like directional policy.
    This policy uses ONLY the goal direction, not the maze walls.
    """
    if np.array_equal(pos, goal_pos):
        return np.ones(4) / 4.0

    v_g = goal_pos - pos
    norm = np.linalg.norm(v_g)
    u_g = v_g / norm

    # Actions in (row, col): UP, DOWN, LEFT, RIGHT
    actions = np.array([
        [-1,  0],  # UP
        [ 1,  0],  # DOWN
        [ 0, -1],  # LEFT
        [ 0,  1]   # RIGHT
    ])

    q_values = np.dot(actions, u_g)
    exp_q = np.exp(kappa * q_values)
    probs = exp_q / np.sum(exp_q)

    return probs


def calculate_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


# =========================
# 3. Simulation (Physics Environment)
# =========================
def run_geo_sim(maze, max_steps=MAX_STEPS):
    pos = np.array([0, 0])
    path = [tuple(pos)]
    actions = []
    entropies = []

    deltas = {
        0: np.array([-1, 0]),  # UP
        1: np.array([1, 0]),   # DOWN
        2: np.array([0, -1]),  # LEFT
        3: np.array([0, 1])    # RIGHT
    }

    for _ in range(max_steps):
        # POLICY: goal-only
        probs = get_geo_probabilities(pos)
        entropy = calculate_entropy(probs)
        action = np.random.choice(4, p=probs)

        # ENVIRONMENT: enforce walls/bounds
        next_pos = pos + deltas[action]

        if (
            0 <= next_pos[0] < maze.shape[0]
            and 0 <= next_pos[1] < maze.shape[1]
            and maze[next_pos[0], next_pos[1]] == 0
        ):
            pos = next_pos
        else:
            # Collision -> wasted step
            pass

        path.append(tuple(pos))
        actions.append(action)
        entropies.append(entropy)

        if np.array_equal(pos, GOAL_POS):
            break

    return np.array(path), np.array(actions), np.array(entropies)


# =========================
# 4. Vector Field Generation
# =========================
def compute_expected_vector_field(maze, kappa=KAPPA):
    """
    Computes E[v] = sum_a P(a|s) * action_vector_a
    for every free cell.
    """
    rows, cols = maze.shape
    U = np.zeros((rows, cols))  # dx (columns)
    V = np.zeros((rows, cols))  # dy (rows)

    # (dx, dy) for plotting
    actions_dxdy = np.array([
        [ 0, -1],  # UP
        [ 0,  1],  # DOWN
        [-1,  0],  # LEFT
        [ 1,  0]   # RIGHT
    ])

    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                continue

            probs = get_geo_probabilities(np.array([r, c]), kappa=kappa)
            expected_v = np.dot(probs, actions_dxdy)

            U[r, c] = expected_v[0]
            V[r, c] = expected_v[1]

    return U, V


# =========================
# 5. Evaluation over many mazes
# =========================
def evaluate_agent(mazes, trials=20, max_steps=MAX_STEPS):
    success = 0
    steps_to_goal = []
    collision_counts = []
    final_distances = []

    total_runs = len(mazes) * trials
    run_count = 0

    for i, maze in enumerate(mazes):
        print(f"Maze {i+1}/{len(mazes)}")

        for t in range(trials):
            run_count += 1
            if run_count % 50 == 0:
                print(f"  Simulation {run_count}/{total_runs}")

            path, _, _ = run_geo_sim(maze, max_steps=max_steps)

            # Success?
            reached_goal = np.array_equal(path[-1], GOAL_POS)
            if reached_goal:
                success += 1
                steps_to_goal.append(len(path) - 1)

            # Useful diagnostics
            unique_moves = np.sum(np.any(np.diff(path, axis=0) != 0, axis=1))
            collisions = (len(path) - 1) - unique_moves
            collision_counts.append(collisions)

            final_dist = np.linalg.norm(path[-1] - GOAL_POS)
            final_distances.append(final_dist)

    success_rate = success / total_runs
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else None
    avg_collisions = np.mean(collision_counts)
    avg_final_distance = np.mean(final_distances)

    return {
        "success_rate": success_rate,
        "avg_steps_to_goal": avg_steps,
        "avg_collisions": avg_collisions,
        "avg_final_distance": avg_final_distance,
        "total_runs": total_runs
    }


# =========================
# 6. Load Maze Dataset
# =========================
print("Loading mazes...")
test_mazes = np.load(DATA_PATH)
print("Loaded mazes:", len(test_mazes))
print("Maze shape:", test_mazes[0].shape)

# Optional sanity checks
assert test_mazes.ndim == 3, "Expected shape: (num_mazes, H, W)"
assert test_mazes.shape[1:] == (16, 16), "Expected 16x16 mazes"
assert test_mazes[0, 0, 0] == 0, "Start cell [0,0] should be free"
assert test_mazes[0, 15, 15] == 0, "Goal cell [15,15] should be free"


# =========================
# 7. Evaluate First
# =========================
print("Starting evaluation...")
results = evaluate_agent(test_mazes, trials=20, max_steps=MAX_STEPS)
print("Evaluation done\n")

print("===== GEO AGENT RESULTS =====")
print(f"Success rate       : {results['success_rate']:.4f}")
print(f"Average steps      : {results['avg_steps_to_goal']}")
print(f"Average collisions : {results['avg_collisions']:.2f}")
print(f"Average final dist : {results['avg_final_distance']:.2f}")
print(f"Total runs         : {results['total_runs']}")


# =========================
# 8. Interactive Viewer
# =========================
current_idx = 0
fig, ax = plt.subplots(figsize=(8, 8))

def update_plot():
    ax.clear()
    maze = test_mazes[current_idx]

    # Background vector field
    U, V = compute_expected_vector_field(maze)
    Y, X = np.mgrid[0:16, 0:16]
    ax.quiver(X, Y, U, -V, color='lightgray', scale=15, width=0.003, headwidth=3)

    # One stochastic rollout
    path, actions, ents = run_geo_sim(maze)

    # Maze
    ax.imshow(maze, cmap='gray_r', extent=[-0.5, 15.5, 15.5, -0.5])

    # Path line
    ax.plot(path[:, 1], path[:, 0], color='blue', linewidth=2, alpha=0.5)

    # Action arrows
    if len(actions) > 0:
        dy = np.array([-1, 1, 0, 0])[actions]
        dx = np.array([0, 0, -1, 1])[actions]
        colors = plt.cm.viridis(ents / np.log(4))  # normalize by max entropy = log(4)
        ax.quiver(
            path[:-1, 1], path[:-1, 0],
            dx, -dy,
            color=colors,
            scale=25,
            width=0.006,
            headwidth=4
        )

    # Start / Goal
    ax.plot(0, 0, 'go', markersize=8, label='Start')
    ax.plot(GOAL_POS[1], GOAL_POS[0], 'r*', markersize=15, label='Goal')

    reached = np.array_equal(path[-1], GOAL_POS)
    title = f"Geocentric Drift | Maze #{current_idx} | κ={KAPPA:.2f}"
    title += "\nSUCCESS" if reached else "\nFAILED / STUCK / WASTED ITS LIFE ON WALLS"

    ax.set_title(title)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(15.5, -0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(test_mazes)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(test_mazes)
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)

print("\nPress Left/Right arrows to switch mazes.")
update_plot()
plt.show()