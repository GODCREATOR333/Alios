import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# --- 1. PARAMETERS ---
GRID_SIZE = 16
N_META_ACTIONS = 2  # 0: Ego, 1: Geo
N_META_STATES = 4608
EPISODES = 150000   # Fast enough for M4
ALPHA = 0.1
GAMMA = 0.99
MAX_STEPS = 250
GOAL = jnp.array([15, 15])

# --- 2. LOAD SUB-POLICIES ---
# Ensure these paths are correct relative to your current folder
try:
    ego_q_sub = jnp.array(np.load("../Alios/Artifacts/EGO_REACTIVE_SURVIVOR_512_policy.npy"))
    geo_q_sub = jnp.array(np.load("../Alios/Artifacts/PURE_GEO_ANGLE_policy.npy"))
    print("✅ Sub-policies loaded.")
except:
    print("❌ Error: Sub-policies not found in Artifacts folder.")
    sys.exit()

# --- 3. JAX CORE LOGIC ---

@jit
def get_meta_state(maze, pos):
    """9-bit window * 9-way compass = 4608 states."""
    r, c = pos
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    win_id = jnp.sum(window.astype(jnp.int32) * jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1]))
    
    dr, dc = jnp.sign(15 - r), jnp.sign(15 - c)
    comp_id = (dr + 1) * 3 + (dc + 1)
    return (win_id * 9 + comp_id).astype(jnp.int32)

@jit
def get_physical_move(maze, pos, meta_choice):
    # Action from Geo (MDP lookup)
    a_geo = jnp.argmax(geo_q_sub[pos[0] * 16 + pos[1]])
    
    # Action from Ego (Window lookup)
    r, c = pos
    padded = jnp.pad(maze, 1, constant_values=1)
    window = lax.dynamic_slice(padded, (r, c), (3, 3)).flatten()
    win_id = jnp.sum(window.astype(jnp.int32) * jnp.array([256, 128, 64, 32, 16, 8, 4, 2, 1]))
    a_ego = jnp.argmax(ego_q_sub[win_id])
    
    return jnp.where(meta_choice == 1, a_geo, a_ego)

@jit
def step_env(maze, pos, meta_choice):
    move = get_physical_move(maze, pos, meta_choice)
    deltas = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    new_pos = pos + deltas[move]
    
    # Physics
    out = (new_pos[0]<0)|(new_pos[0]>=16)|(new_pos[1]<0)|(new_pos[1]>=16)
    safe_rc = jnp.clip(new_pos, 0, 15)
    hit = out | (maze[safe_rc[0], safe_rc[1]] == 1)
    
    actual_pos = jnp.where(hit, pos, new_pos)
    is_goal = (actual_pos[0] == 15) & (actual_pos[1] == 15)
    
    # --- DUNG BEETLE REWARD SHAPING ---
    old_dist = jnp.sum(jnp.abs(pos - GOAL))
    new_dist = jnp.sum(jnp.abs(actual_pos - GOAL))
    # Aggressive multiplier (2.0) creates strong gravity toward the goal
    progress_reward = (old_dist - new_dist) * 2.0 - 0.2
    
    reward = jnp.where(is_goal, 100.0, jnp.where(hit, -10.0, progress_reward))
    return actual_pos, reward, is_goal

# --- 4. TRAINING ENGINE ---

@jit
def train_episode(carry, episode_idx):
    q_table, rng, mazes, eps_decay = carry
    key, k_maze, k_l = jax.random.split(rng, 3)
    maze = mazes[jax.random.randint(k_maze, (), 0, mazes.shape[0])]
    epsilon = jnp.maximum(0.01, 1.0 * (eps_decay ** episode_idx))

    def body_fun(state):
        p, q, stp, done, rew_acc, k = state
        k, k1, k2 = jax.random.split(k, 3)
        sid = get_meta_state(maze, p)
        a_meta = jnp.where(jax.random.uniform(k1) < epsilon, 
                          jax.random.randint(k2, (), 0, 2), 
                          jnp.argmax(q[sid]))
        
        next_p, rew, is_done = step_env(maze, p, a_meta)
        nsid = get_meta_state(maze, next_p)
        
        target = rew + GAMMA * jnp.where(is_done, 0.0, jnp.max(q[nsid]))
        q = q.at[sid, a_meta].add(ALPHA * (target - q[sid, a_meta]))
        return next_p, q, stp + 1, is_done | (stp > MAX_STEPS), rew_acc + rew, k

    init = (jnp.array([0,0]), q_table, 0, False, 0.0, k_l)
    final = lax.while_loop(lambda s: ~s[3], body_fun, init)
    success = jnp.where((final[0][0] == 15) & (final[0][1] == 15), 1.0, 0.0)
    return (final[1], key, mazes, eps_decay), success

# --- 5. EXECUTION ---

print("🚀 Training Hierarchical Meta-Switcher...")
train_data = jnp.array(np.load("data_jax/N16_P0100_train_solvable.npy"))
q_init = jnp.zeros((N_META_STATES, N_META_ACTIONS))
# Initialize with small optimistic value to encourage exploring both modes
q_init = q_init + 5.0 

start_t = time.time()
(final_q, _, _, _), successes = lax.scan(train_episode, (q_init, jax.random.PRNGKey(42), train_data, 0.9999), jnp.arange(EPISODES))
print(f"✅ Training Done in {time.time()-start_t:.2f}s")
print(f"📊 Final success rate in training: {jnp.mean(successes[-1000:])*100:.1f}%")

# --- 6. INTERACTIVE VISUALIZER ---

test_mazes = np.load("data_jax/N16_P0100_test_solvable_random.npy")
current_idx = 0

def run_sim(idx):
    maze = test_mazes[idx]
    pos, path, modes = jnp.array([0, 0]), [(0,0)], []
    for _ in range(MAX_STEPS):
        sid = get_meta_state(maze, pos)
        mode = int(np.argmax(final_q[sid])) # 0: Ego, 1: Geo
        move = int(get_physical_move(maze, pos, mode))
        pos, _, done = step_env(maze, pos, mode)
        path.append(tuple(map(int, pos)))
        modes.append(mode)
        if done: break
    return np.array(path), np.array(modes)

fig, ax = plt.subplots(figsize=(7,7))

def update():
    ax.clear()
    path, modes = run_sim(current_idx)
    ax.imshow(test_mazes[current_idx], cmap='gray_r')
    for i in range(len(path)-1):
        color = 'cyan' if modes[i] == 0 else 'red' # Cyan=Ego, Red=Geo
        ax.plot(path[i:i+2, 1], path[i:i+2, 0], color=color, linewidth=2, alpha=0.8)
    success = "✓" if (path[-1] == [15,15]).all() else "✗"
    ax.set_title(f"Meta-Switcher | Maze {current_idx} | {success}\nRED: Geo (Magnet) | CYAN: Ego (Survivor)")
    plt.draw()

def on_key(event):
    global current_idx
    if event.key == 'right': current_idx = (current_idx + 1) % 100
    elif event.key == 'left': current_idx = (current_idx - 1) % 100
    update()

fig.canvas.mpl_connect('key_press_event', on_key)
update()
print("\n🎮 USE ARROW KEYS TO EXPLORE MAZES")
plt.show()