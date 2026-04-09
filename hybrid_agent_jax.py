import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import db_manager

# =========================================================
# 1. PHYSICS & RL CONFIGURATION
# =========================================================
KAPPA = 2.5            # Goal bias strength (GEO)
D_R = 0.15             # Rotational diffusion (EGO)
P_STRAIGHT = jnp.exp(-D_R)
P_TURN = (1.0 - P_STRAIGHT) / 2.0

# State: 256 (window) * 9 (goal_dir) * 5 (last_action) = 11,520
Q_TABLE_SIZE = 11520 
TRAIN_EPISODES = 80000 
ALPHA, GAMMA = 0.1, 0.98
EPS_START, EPS_MIN, EPS_DECAY = 1.0, 0.05, 0.99995

# =========================================================
# 2. PHYSICS & ENVIRONMENT ENGINES (JAX JIT)
# =========================================================

@jax.jit
def get_meta_state(maze, r, c, last_action):
    padded = jnp.pad(maze, 1, constant_values=1)
    window = jax.lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    neighbors = jnp.concatenate([window[:4], window[5:]])
    window_id = jnp.sum(neighbors * jnp.array([128, 64, 32, 16, 8, 4, 2, 1]))
    dr_sign, dc_sign = jnp.sign(15 - r) + 1, jnp.sign(15 - c) + 1
    goal_id = dr_sign * 3 + dc_sign
    return (window_id * 45 + goal_id * 5 + last_action).astype(jnp.int32)

@jax.jit
def get_geo_probs(r, c):
    vr, vc = 15.0 - r, 15.0 - c
    norm = jnp.sqrt(vr**2 + vc**2) + 1e-5
    q = jnp.array([-1, 1, 0, 0])*(vr/norm) + jnp.array([0, 0, -1, 1])*(vc/norm)
    return jax.nn.softmax(KAPPA * q)

@jax.jit
def get_ego_probs(maze, r, c, last_a):
    padded = jnp.pad(maze, 1, constant_values=1)
    is_w = jnp.array([padded[r, c+1], padded[r+2, c+1], padded[r+1, c], padded[r+1, c+2]])
    base = jnp.where(last_a == 4, jnp.full(4, 0.25), jnp.full(4, P_TURN))
    safe_la = jnp.clip(last_a, 0, 3)
    base = jnp.where(last_a < 4, base.at[safe_la].set(P_STRAIGHT), base)
    opp = jnp.array([1, 0, 3, 2])[safe_la]
    base = jnp.where(last_a < 4, base.at[opp].set(0.0), base)
    masked = jnp.where(is_w == 1, 0.0, base)
    s = jnp.sum(masked)
    return jnp.where(s > 0, masked / (s + 1e-8), jnp.zeros(4).at[opp].set(1.0))

@jax.jit
def env_step(maze, r, c, last_a, meta_a, key):
    probs = jnp.where(meta_a == 0, get_geo_probs(r, c), get_ego_probs(maze, r, c, last_a))
    key, subkey = jrandom.split(key)
    a = jrandom.choice(subkey, 4, p=probs)
    nr, nc = r + jnp.array([-1, 1, 0, 0])[a], c + jnp.array([0, 0, -1, 1])[a]
    invalid = (nr < 0) | (nr >= 16) | (nc < 0) | (nc >= 16) | (maze[jnp.clip(nr,0,15), jnp.clip(nc,0,15)] == 1)
    fr, fc = jnp.where(invalid, r, nr), jnp.where(invalid, c, nc)
    is_goal = (fr == 15) & (fc == 15)
    reward = jnp.where(is_goal, 100.0, jnp.where(invalid, -5.0, -1.0))
    return fr, fc, a, reward, is_goal, key

# =========================================================
# 3. TRAINING & BATCH EVALUATION
# =========================================================

@jax.jit
def train_step(q, mazes, eps, k):
    k, sk1, sk2 = jrandom.split(k, 3)
    maze = mazes[jrandom.randint(sk1, (), 0, len(mazes))]
    def body(s):
        q, r, c, la, st, done, k = s
        state = get_meta_state(maze, r, c, la)
        k, sk = jrandom.split(k)
        meta_a = jnp.where(jrandom.uniform(sk) < eps, jrandom.randint(sk, (), 0, 2), jnp.argmax(q[state]))
        nr, nc, na, rew, d, k = env_step(maze, r, c, la, meta_a, k)
        target = rew + GAMMA * jnp.where(d, 0.0, jnp.max(q[get_meta_state(maze, nr, nc, na)]))
        q = q.at[state, meta_a].set(q[state, meta_a] + ALPHA * (target - q[state, meta_a]))
        return (q, nr, nc, na, st + 1, d, k)
    return jax.lax.while_loop(lambda s: (~s[5]) & (s[4] < 300), body, (q, 0, 0, 4, 0, False, k))[0], k

@jax.jit
def evaluate_batch(mazes, q, keys):
    def run_one(maze, k):
        def body(s):
            r, c, la, st, ego_c, d, k = s
            state = get_meta_state(maze, r, c, la)
            meta_a = jnp.argmax(q[state])
            nr, nc, na, _, d, k = env_step(maze, r, c, la, meta_a, k)
            return (nr, nc, na, st + 1, ego_c + meta_a, d, k)
        res = jax.lax.while_loop(lambda s: (~s[5]) & (s[4] < 300), body, (0, 0, 4, 0, 0, False, k))
        return res[5], res[3], res[4] 
    return jax.vmap(run_one)(mazes, keys)

# =========================================================
# 4. EXECUTION: TRAINING & DATA LOADING
# =========================================================
print("Loading datasets...")
train_data = jnp.array(np.load("data_jax/N16_P0100_train_solvable.npy"))
test_rand  = jnp.array(np.load("data_jax/N16_P0100_test_solvable_random.npy"))
test_shapes = jnp.array(np.load("data_jax/N16_P0100_test_solvable_shapes.npy"))
test_symm   = jnp.array(np.load("data_jax/N16_P0100_test_solvable_symmetric.npy"))

# q_table = jnp.zeros((Q_TABLE_SIZE, 2))
q_table = jnp.full((Q_TABLE_SIZE, 2), -50.0)
rng_key = jrandom.PRNGKey(42)

print(f"Training on {len(train_data)} mazes...")
t0 = time.perf_counter()
for ep in range(TRAIN_EPISODES):
    curr_eps = max(EPS_MIN, EPS_START * (EPS_DECAY**ep))
    q_table, rng_key = train_step(q_table, train_data, curr_eps, rng_key)
print(f"Training complete in {time.perf_counter()-t0:.2f}s")

# Global results for the viewer
current_maze_idx = 0
active_set = test_rand

# =========================================================
# 5. INTERACTIVE VIEWER
# =========================================================
fig, ax = plt.subplots(figsize=(9, 9))

def update_plot():
    global rng_key
    ax.clear()
    maze = active_set[current_maze_idx]
    
    # Run a greedy rollout for visualization
    def rollout():
        r, c, la, st = 0, 0, 4, 0
        path, modes, actions = [(0,0)], [], []
        k = rng_key
        for _ in range(400):
            state = get_meta_state(maze, r, c, la)
            meta_a = int(jnp.argmax(q_table[state]))
            nr, nc, na, _, d, k = env_step(maze, r, c, la, meta_a, k)
            path.append((int(nr), int(nc)))
            modes.append(meta_a)
            actions.append(int(na))
            r, c, la = nr, nc, na
            if d: break
        return np.array(path), np.array(modes), np.array(actions)

    path, modes, actions = rollout()
    
    # Draw Maze
    ax.imshow(maze, cmap='gray_r', alpha=0.2)
    
    # Draw Trajectory
    for i in range(len(modes)):
        color = 'blue' if modes[i] == 0 else 'red'
        ax.plot(path[i:i+2, 1], path[i:i+2, 0], color=color, lw=2.5, alpha=0.7)
        
        # Action Arrows
        dr = [-0.3, 0.3, 0, 0][actions[i]]
        dc = [0, 0, -0.3, 0.3][actions[i]]
        ax.arrow(path[i, 1], path[i, 0], dc, dr, head_width=0.2, head_length=0.2, fc=color, ec=color)

    # Markers
    ax.plot(0, 0, 'go', ms=10)
    ax.plot(15, 15, 'r*', ms=20)
    
    # Metrics Text
    succ = np.array_equal(path[-1], [15,15])
    ego_pct = (np.sum(modes==1)/len(modes))*100 if len(modes)>0 else 0
    ax.set_title(f"Maze #{current_maze_idx} | {'SUCCESS' if succ else 'FAILED'}\nSteps: {len(modes)} | EGO Usage: {ego_pct:.1f}%", fontsize=14)
    
    # Legend
    legend_elements = [Line2D([0], [0], color='blue', lw=3, label='GEO (Drift)'),
                       Line2D([0], [0], color='red', lw=3, label='EGO (Sliding)')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_xticks([]); ax.set_yticks([])
    plt.draw()

def on_key(event):
    global current_maze_idx
    if event.key == 'right':
        current_maze_idx = (current_maze_idx + 1) % len(active_set)
    elif event.key == 'left':
        current_maze_idx = (current_maze_idx - 1) % len(active_set)
    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
print("\nInteractive Viewer Ready. Use Left/Right arrows to browse mazes.")

# Run initial evaluations for terminal output
for name, m_set in [
    ("Random", test_rand),
    ("Shapes", test_shapes),
    ("Symmetric", test_symm)
]:
    rng_key, subkey = jrandom.split(rng_key)
    keys = jrandom.split(subkey, len(m_set))
    succ, steps, ego = evaluate_batch(m_set, q_table, keys)
    mfpt = jnp.mean(steps[succ]) if jnp.any(succ) else float('nan')
    print(f"[{name:10}] SR: {jnp.mean(succ)*100:5.2f}% | MFPT: {mfpt:.2f}" if not jnp.isnan(mfpt) else f"[{name:10}] SR: 0.00% | MFPT: N/A")

update_plot()
plt.show()

# 1. Prepare Metadata
run_id = "META_RL_V1_p10"
config = {
    "kappa": 2.5,
    "dr": 0.15,
    "alpha": 0.1,
    "gamma": 0.98,
    "episodes": 80000,
    "description": "Learned Meta-Policy switching between GEO/EGO-- Trained on 10% Density"
}

# 2. Register it (This saves the .npy to Artifacts/ and updates the DB)
# Note: We use 'meta_11k' as the state_repr key
db_manager.register_run(
    run_id=run_id, 
    algo="Meta-Q-Learning", 
    state_repr="meta_11k", 
    config_dict=config, 
    q_table=np.array(q_table) # Ensure it's a numpy array for saving
)

print(f"Successfully registered {run_id} to Alios Database.")