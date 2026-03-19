import os
import json
import numpy as np
import jax.numpy as jnp
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

# Local Imports
from viewer_2d import MazeView
import db_manager
import core_logic

class AliosWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # --- 1. Window Meta ---
        self.setWindowTitle("ALIOS: Mechanistic Interpreter")
        self.resize(1400, 900)

        # --- 2. Data State (The Internal 'Backpack') ---
        self.current_mazes = None          # Raw .npy maze dataset
        self.current_vi_policies = None    # The Oracle baseline
        self.current_agent_q = None        # The Loaded Q-table/Policy
        self.current_decoder = None        # Function to map (r,c) -> State ID
        self.current_config = None         # JSON metadata for the run
        self.current_vi_values = None

        # --- 3. Initialize UI & Connections ---
        self.init_ui()
        self.connect_signals()

        # --- 4. Startup Sequences ---
        self.refresh_runs()
        self.scan_datasets()

    def init_ui(self):
        """Builds the layout and widgets (Modular approach)."""
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ============================
        # SIDEBAR (LEFT)
        # ============================
        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar.setMinimumWidth(300)

        # --- Run/Artifact Selection ---
        self.run_group = QtWidgets.QGroupBox("Artifact Registry (Runs)")
        run_layout = QtWidgets.QVBoxLayout(self.run_group)
        self.run_selector = QtWidgets.QComboBox()
        self.refresh_runs_btn = QtWidgets.QPushButton("Refresh Database")
        run_layout.addWidget(self.run_selector)
        run_layout.addWidget(self.refresh_runs_btn)
        self.sidebar_layout.addWidget(self.run_group)

        # --- Dataset Selection ---
        self.data_group = QtWidgets.QGroupBox("Test Dataset")
        data_layout = QtWidgets.QVBoxLayout(self.data_group)
        self.dataset_selector = QtWidgets.QComboBox()
        data_layout.addWidget(self.dataset_selector)
        self.sidebar_layout.addWidget(self.data_group)

        # --- Maze Navigation (Slider + Number Box) ---
        self.maze_group = QtWidgets.QGroupBox("Maze Navigator")
        maze_nav_layout = QtWidgets.QVBoxLayout(self.maze_group)
        
        top_nav_layout = QtWidgets.QHBoxLayout()
        self.maze_label = QtWidgets.QLabel("Maze Index:")
        
        self.maze_spinbox = QtWidgets.QSpinBox()
        self.maze_spinbox.setRange(0, 999)
        self.maze_spinbox.setFixedWidth(60)
        
        # --- THE FIX: Hide the black up/down UI arrows ---
        self.maze_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.maze_spinbox.setAlignment(QtCore.Qt.AlignCenter) # Center the number
                    
        top_nav_layout.addWidget(self.maze_label)
        top_nav_layout.addSpacing(5)
        top_nav_layout.addWidget(self.maze_spinbox)
        top_nav_layout.addStretch()
        
        self.maze_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maze_slider.setRange(0, 999)
        self.maze_spinbox.setStyleSheet("""
                QSpinBox {
                    padding: 4px;
                    border-radius: 6px;
                    border: 1px solid #444;
                    background-color: #1e1e1e;
                    color: #ddd;
                }
            """)
        
        self.maze_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background: #2a2a2a;
                    border-radius: 3px;
                }

                QSlider::handle:horizontal {
                    background: #00BFFF;
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }

                QSlider::sub-page:horizontal {
                    background: #00BFFF;
                    border-radius: 3px;
                }
            """)
        
        self.maze_group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #333;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding: 8px;
                }

                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 3px;
                }
            """)
        
        self.maze_slider.setToolTip("Scroll through maze samples")
        self.maze_spinbox.setToolTip("Enter maze index")
        maze_nav_layout.setContentsMargins(10, 12, 10, 12)
        maze_nav_layout.setSpacing(10)
        top_nav_layout.setSpacing(8)

        # Use setTracking(False) if you want it to ONLY update when you release the mouse, 
        # but we'll keep it True and optimize the loop instead.
        
        maze_nav_layout.addLayout(top_nav_layout)
        maze_nav_layout.addWidget(self.maze_slider)
        self.sidebar_layout.addWidget(self.maze_group)


        # --- Stats Group ---
        self.stats_group = QtWidgets.QGroupBox("Statistical Performance Profile")
        stats_layout = QtWidgets.QVBoxLayout(self.stats_group)
        
        self.run_test_btn = QtWidgets.QPushButton("Run Batch Test (1000 Mazes)")
        self.run_test_btn.setMinimumHeight(40)
        self.run_test_btn.setStyleSheet("background-color: #28A745; color: white; font-weight: bold; border-radius: 5px;")
        
        self.stats_display = QtWidgets.QLabel("Select a dataset and run test...")
        self.stats_display.setStyleSheet("""
            font-family: 'Courier New'; 
            font-size: 10pt; 
            color: #00FF00; 
            background-color: #000; 
            padding: 8px;
        """)
        
        stats_layout.addWidget(self.run_test_btn)
        stats_layout.addWidget(self.stats_display)
        self.sidebar_layout.insertWidget(2, self.stats_group)


        # --- Micro-Inspector (Neuro-Probe) ---
        self.inspect_group = QtWidgets.QGroupBox("Neuro-Probe: Q-Values")
        inspect_layout = QtWidgets.QVBoxLayout(self.inspect_group)
        
        # Replace PlotWidget with a rich-text QLabel
        self.neuro_label = QtWidgets.QLabel("Click on any cell in the Agent View to probe its internal state.")
        self.neuro_label.setStyleSheet("""
            font-family: 'Courier New', monospace; 
            font-size: 11pt; 
            color: #00BFFF;
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 5px;
        """)
        self.neuro_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        inspect_layout.addWidget(self.neuro_label)
        self.sidebar_layout.addWidget(self.inspect_group)

        # --- Probes Info Legend ---
        self.probes_help = QtWidgets.QGroupBox("Probes Guide")
        help_layout = QtWidgets.QVBoxLayout(self.probes_help)
        self.probes_help.setStyleSheet("QGroupBox { color: #FFD700; }") # Gold title
        
        help_text = QtWidgets.QLabel(
            "<b>• CLICK:</b> Run Micro-Probe on cell.<br>"
            "<b>• YELLOW BOX:</b> Aliased states (Confusion).<br>"
            "<b>• CONFLICT:</b> Mathematical solvability.<br>"
            "<b>• ENTROPY:</b> Policy uncertainty (Ties).<br>"
            "<b>• PATH:</b> Visualizes Limit Cycles (Loops)."
        )
        help_text.setStyleSheet("font-size: 9pt; color: #BBB;")
        help_layout.addWidget(help_text)
        self.sidebar_layout.addWidget(self.probes_help)


        # --- Config/Metadata Inspector ---
        self.config_group = QtWidgets.QGroupBox("Run Configuration (JSON)")
        config_layout = QtWidgets.QVBoxLayout(self.config_group)
        self.config_display = QtWidgets.QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setStyleSheet("""
            background-color: #1e1e1e; 
            color: #d4d4d4; 
            font-family: 'Courier New'; 
            font-size: 10pt;
        """)
        config_layout.addWidget(self.config_display)
        self.sidebar_layout.addWidget(self.config_group)

        self.sidebar_layout.addStretch()

        # ============================
        # VIEWER PANELS (RIGHT)
        # ============================
        self.viewer_widget = QtWidgets.QWidget()
        self.viewer_layout = QtWidgets.QHBoxLayout(self.viewer_widget)
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- NEW: Oracle Tab Container ---
        self.oracle_tabs = QtWidgets.QTabWidget()
        self.oracle_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; top: -1px; background: #121212; }
            QTabBar::tab { background: #2a2a2a; color: #aaa; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #121212; color: #00BFFF; border: 1px solid #333; border-bottom: none; }
        """)

        # Tab 1: Policy View (Reuse your MazeView)
        self.view_oracle_policy = MazeView(title="VI: Optimal Policy")
        
        # Tab 2: Value View (Heatmap)
        self.view_oracle_values = MazeView(title="VI: State Values (V*)")
        
        self.oracle_tabs.addTab(self.view_oracle_policy, "Policy (π*)")
        self.oracle_tabs.addTab(self.view_oracle_values, "Value Heatmap (V*)")

        # --- NEW: Agent Tab Container ---
        self.agent_tabs = QtWidgets.QTabWidget()
        self.agent_tabs.setStyleSheet(self.oracle_tabs.styleSheet()) # Reuse the exact same CSS!

        self.view_agent_policy = MazeView(title="Agent: Learned Policy")
        self.view_agent_values = MazeView(title="Agent: Perceived Values (max Q)")
        self.view_agent_entropy = MazeView(title="Agent: Policy Entropy (Confusion)")
        
        self.agent_tabs.addTab(self.view_agent_policy, "Policy (π)")
        self.agent_tabs.addTab(self.view_agent_values, "Value Heatmap (max Q)")
        self.agent_tabs.addTab(self.view_agent_entropy, "Entropy (H)")
        
        # Add both Tab Containers to the Splitter
        self.viewer_splitter.addWidget(self.oracle_tabs)
        self.viewer_splitter.addWidget(self.agent_tabs)
        self.viewer_splitter.setSizes([800, 800])

        self.viewer_layout.addWidget(self.viewer_splitter)

        # Assemble Main Splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.viewer_widget)
        self.splitter.setSizes([300, 1100])
        self.main_layout.addWidget(self.splitter)


        # Agent View (Stays the same)
        self.view_agent = MazeView(title="Agent Policy (Hypothesis)")


    def connect_signals(self):
        """Connects UI elements to logic methods."""
        self.refresh_runs_btn.clicked.connect(self.refresh_runs)
        self.run_selector.currentIndexChanged.connect(self.on_run_selected)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_selected)
        self.view_agent_entropy.cellClicked.connect(self.update_neuro_probe)
        
        # --- SYNC SLIDER AND SPINBOX ---
        self.maze_slider.valueChanged.connect(self.maze_spinbox.setValue)
        self.maze_spinbox.valueChanged.connect(self.maze_slider.setValue)
        
        # Both trigger the display update via the slider
        self.maze_slider.valueChanged.connect(self.update_display)

        # Run Batch Test
        self.run_test_btn.clicked.connect(self.run_batch_test)

        # Connect Neuro-Probe
        self.view_agent.cellClicked.connect(self.update_neuro_probe)
        # Connect Neuro-Probe to BOTH Agent views so you can click the heatmap too!
        self.view_agent_policy.cellClicked.connect(self.update_neuro_probe)
        self.view_agent_values.cellClicked.connect(self.update_neuro_probe)
    
    def keyPressEvent(self, event):
        """Allows global keyboard arrow keys to scrub through mazes."""
        if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_Right:
            self.maze_slider.setValue(self.maze_slider.value() + 1)
        elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_Left:
            self.maze_slider.setValue(self.maze_slider.value() - 1)
        else:
            super().keyPressEvent(event)

    # ============================
    # DATA & LOGIC METHODS
    # ============================

    def refresh_runs(self):
        """Queries the SQLite DB via db_manager and updates the run dropdown."""
        self.run_selector.blockSignals(True) # Prevent triggering on_run_selected while loading
        self.run_selector.clear()
        runs = db_manager.get_all_runs()
        self.run_selector.addItems(runs)
        self.run_selector.blockSignals(False)

    def scan_datasets(self):
        """Lists all .npy maze files and picks the first one."""
        self.dataset_selector.blockSignals(True)
        path = "data_jax"
        if os.path.exists(path):
            files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
            self.dataset_selector.addItems(files)
        self.dataset_selector.blockSignals(False)
        
        # Manually trigger the first load
        if self.dataset_selector.count() > 0:
            self.dataset_selector.setCurrentIndex(0)
            self.on_dataset_selected()


    def on_run_selected(self):
        """Triggered when a user picks a different training run."""
        run_id = self.run_selector.currentText()
        if not run_id: return
        
        details = db_manager.get_run_details(run_id)
        if details:
            repr_key = details['state_repr']
            # Use .get(key, default) to prevent NoneType crashes!
            self.current_decoder = core_logic.DECODERS.get(
                repr_key, core_logic.decode_mdp
            )
            
            self.vector_decoder_jit = core_logic.STATE_MAP_FUNCS_JIT.get(
                repr_key, core_logic.get_full_state_map_mdp
            )

    
            # --- THE FIX: Create both versions ---
            # JIT version is for the slider (buttery smooth)
            self.vector_decoder_jit = core_logic.STATE_MAP_FUNCS_JIT.get(repr_key)
            # RAW version is for the Batch Evaluator (prevents JAX crash)
            self.vector_decoder_raw = core_logic.STATE_MAP_FUNCS_RAW.get(repr_key)
            
            # Update display
            self.config_display.setText(json.dumps(details['config'], indent=4))
            self.current_agent_q = jnp.array(np.load(details['path'])) # Ensure Q is JAX array
            
            self.update_display()

    def on_dataset_selected(self):
        ds_name = self.dataset_selector.currentText()
        if not ds_name: return
        
        # 1. Load raw mazes
        self.current_mazes = np.load(os.path.join("data_jax", ds_name))
        self.current_mazes_jax = jnp.array(self.current_mazes)
        
        # 2. Load Oracle .npz
        vi_filename = ds_name.replace(".npy", "_VI_solved.npz")
        vi_path = os.path.join("data_jax", "value_iteration", vi_filename)
        
        if os.path.exists(vi_path):
            with np.load(vi_path) as data:
                self.current_vi_policies = data['policy']
                self.current_vi_values = data['values'] # <--- ADD THIS LINE
            print(f"Loaded Oracle: {vi_filename}")
        else:
            self.current_vi_policies = None
            self.current_vi_values = None # <--- ADD THIS LINE

        self.update_display()

    def on_maze_slider_moved(self):
        """Update label and redraw views when slider moves."""
        idx = self.maze_slider.value()
        self.maze_label.setText(f"Maze Index: {idx}")
        self.update_display()


    def run_batch_test(self):
        if self.current_agent_q is None or self.current_mazes_jax is None:
            return
            
        self.run_test_btn.setText("Computing Rollouts...")
        QtWidgets.QApplication.processEvents()
        
        # 1. Run JAX Rollouts
        res = core_logic.evaluate_dataset(
            self.current_agent_q, 
            self.current_mazes_jax, 
            self.vector_decoder_raw
        )
        
        # Unpack results from JAX
        # (r, c, steps, collisions, reached_goal)
        steps_arr = np.array(res[2])
        colls_arr = np.array(res[3])
        goal_arr = np.array(res[4])
        
        # 2. Basic Stats
        total = len(goal_arr)
        success_count = np.sum(goal_arr)
        success_rate = (success_count / total) * 100
        timeout_rate = (np.sum(steps_arr >= 500) / total) * 100
        
        # 3. Efficiency Stats (Successful runs only)
        avg_steps = 0
        avg_colls = 0
        opt_gap = 1.0
        
        if success_count > 0:
            success_mask = (goal_arr == True)
            avg_steps = np.mean(steps_arr[success_mask])
            avg_colls = np.mean(colls_arr[success_mask])
            
            # --- THE OPTIMALITY GAP ---
            if self.current_vi_policies is not None:
                # Calculate Oracle's average steps for the SAME successful mazes
                # (Remember: VI values at start (0,0) are -steps to goal)
                # We use absolute because values are negative
                oracle_steps_all = np.abs(self.current_vi_values[:, 0, 0])
                avg_oracle_steps = np.mean(oracle_steps_all[success_mask])
                opt_gap = avg_steps / avg_oracle_steps

        # 4. Display Result
        self.stats_display.setText(
            f"<b>RESULT SUMMARY:</b><br>"
            f"-------------------------<br>"
            f"Success Rate : {success_rate:>6.1f}%<br>"
            f"Timeout Rate : {timeout_rate:>6.1f}%<br>"
            f"Avg Collide  : {avg_colls:>6.1f}<br>"
            f"Avg Steps    : {avg_steps:>6.1f}<br>"
            f"-------------------------<br>"
            f"<b style='color: #FFD700;'>Optimality Gap: {opt_gap:.2f}x</b>"
        )
        self.run_test_btn.setText("Run Batch Test (1000 Mazes)")


    def update_neuro_probe(self, r, c, state_id, aliased_count):
        if self.current_agent_q is not None:
            q_vals = self.current_agent_q[state_id]
            
            # 1. Calculate Entropy (Confusion)
            entropy = core_logic.calculate_entropy(q_vals)
            
            # 2. Calculate Conflict (Fatal vs Benign Aliasing)
            # We need the current state_map and oracle_policy
            idx = self.maze_slider.value()
            state_map = self.vector_decoder_jit(self.current_mazes_jax[idx])
            oracle_p = self.current_vi_policies[idx]
            
            conflict = core_logic.calculate_conflict(state_id, state_map, oracle_p)
            
            # 3. Build the Rich Text Readout
            color_id = "#FFD700" # Gold
            text = (
                f"<b style='color:{color_id};'>State ID:</b> {state_id}<br>"
                f"<b style='color:{color_id};'>Aliased:</b> {aliased_count} locations<br>"
                f"<b style='color:{color_id};'>Conflict:</b> {conflict:.1f}%<br>"
                f"<b style='color:{color_id};'>Entropy:</b> {entropy:.3f}<br>"
                f"<hr style='border: 1px solid #444;'>"
                f"<b>[↑] Up   :</b> {q_vals[0]:>8.2f}<br>"
                f"<b>[↓] Down :</b> {q_vals[1]:>8.2f}<br>"
                f"<b>[←] Left :</b> {q_vals[2]:>8.2f}<br>"
                f"<b>[→] Right:</b> {q_vals[3]:>8.2f}<br>"
            )
            
            self.neuro_label.setText(text)
            self.inspect_group.setTitle(f"Neuro-Probe: ({r}, {c})")

            # --- NEW: Compute and Draw Trajectory ---
            idx = self.maze_slider.value()
            maze = self.current_mazes[idx]
            
            # Compute path using the scalar decoder
            path = core_logic.compute_rollout(
                maze, 
                (r, c), 
                self.current_agent_q, 
                self.current_decoder
            )
            
            # Draw the line on both Agent tabs
            self.view_agent_policy.draw_trajectory(path)
            self.view_agent_values.draw_trajectory(path)

        
    def update_display(self):
        idx = self.maze_slider.value()
        if self.current_mazes is None:
            return
        
        maze = self.current_mazes[idx]
        maze_jax_slice = self.current_mazes_jax[idx]

        # Reset all views
        self.view_oracle_policy.set_maze(maze)
        self.view_oracle_values.set_maze(maze)
        self.view_agent_policy.set_maze(maze)
        self.view_agent_values.set_maze(maze)
        self.view_agent_entropy.set_maze(maze)

        # --- ORACLE TAB ---
        if self.current_vi_policies is not None and self.current_vi_values is not None:
            self.view_oracle_policy.draw_policy_vectorized(
                maze,
                self.current_vi_policies[idx],
                base_color='#28A745'
            )
            self.view_oracle_values.set_heatmap(maze, self.current_vi_values[idx])

        # --- AGENT TAB ---
        if self.current_agent_q is not None and callable(getattr(self, 'vector_decoder_jit', None)):
            
            state_id_map = self.vector_decoder_jit(maze_jax_slice)

            # Q-vectors per state: (16, 16, 4)
            q_vectors = self.current_agent_q[state_id_map]

            # Metrics
            agent_actions = np.argmax(q_vectors, axis=-1)
            agent_values = np.max(q_vectors, axis=-1)

            # Entropy (NEW)
            agent_entropy = core_logic.calculate_entropy_grid(q_vectors)

            # Provide state map to UI
            self.view_agent_policy.set_state_map(state_id_map)
            self.view_agent_values.set_state_map(state_id_map)
            self.view_agent_entropy.set_state_map(state_id_map)

            oracle_actions = self.current_vi_policies[idx] if self.current_vi_policies is not None else None

            # Draw policy
            self.view_agent_policy.draw_policy_vectorized(
                maze,
                agent_actions,
                oracle_actions=oracle_actions
            )

            # Draw value heatmap
            self.view_agent_values.set_heatmap(maze, agent_values)

            # Draw entropy heatmap
            self.view_agent_entropy.set_heatmap(maze, agent_entropy)