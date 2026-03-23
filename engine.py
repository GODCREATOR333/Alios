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
        self.current_mazes = None          
        self.current_mazes_jax = None
        self.current_vi_policies = None    
        self.current_vi_values = None
        self.current_maze_score = ""

        # --- NEW: Symmetrical Panel States ---
        self.left_type = 'oracle'  # 'oracle' or 'agent'
        self.left_q = None
        self.left_decoder = None
        self.left_decoder_jit = None
        self.left_decoder_raw = None 
        self.left_rollout_cache = None

        self.right_type = 'agent'
        self.right_q = None
        self.right_decoder = None
        self.right_decoder_jit = None
        self.right_decoder_raw = None 
        self.right_rollout_cache = None 

        # --- 3. Initialize UI & Connections ---
        self.init_ui()
        self.connect_signals()

        # --- 4. Startup Sequences ---
        self.refresh_runs()
        self.scan_datasets()

    def init_ui(self):
        """Builds the layout and widgets (Modular approach)."""
        self.main_container = QtWidgets.QVBoxLayout(self)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        self.main_container.setSpacing(0)

        # --- 1. WORKSPACE SWITCHER HEADER ---
        self.header = QtWidgets.QWidget()
        self.header.setFixedHeight(50)
        self.header.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")
        header_layout = QtWidgets.QHBoxLayout(self.header)
        
        self.btn_inspector = QtWidgets.QPushButton("DEEP INSPECTOR")
        self.btn_analytics = QtWidgets.QPushButton("ANALYTICS LAB")
        
        btn_style = """
            QPushButton { background-color: transparent; color: #888; font-weight: bold; padding: 10px 20px; border: None; font-size: 10pt; }
            QPushButton:checked { color: #00BFFF; border-bottom: 2px solid #00BFFF; }
            QPushButton:hover { color: #DDD; }
        """
        self.workspace_group = QtWidgets.QButtonGroup(self)
        for i, btn in enumerate([self.btn_inspector, self.btn_analytics]):
            btn.setStyleSheet(btn_style)
            btn.setCheckable(True)
            header_layout.addWidget(btn)
            self.workspace_group.addButton(btn, i)
        
        self.btn_inspector.setChecked(True)
        header_layout.addStretch()
        self.main_container.addWidget(self.header)

        # --- 2. STACKED WORKSPACE CONTAINER ---
        self.workspaces = QtWidgets.QStackedWidget()
        self.main_container.addWidget(self.workspaces)

        # --- WORKSPACE 1: DEEP INSPECTOR ---
        self.inspector_page = QtWidgets.QWidget()
        self.inspector_layout = QtWidgets.QHBoxLayout(self.inspector_page)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ============================
        # SIDEBAR (LEFT)
        # ============================
        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar.setMinimumWidth(300)

        # --- NEW: Symmetrical Source Selection ---
        self.source_group = QtWidgets.QGroupBox("Panel Sources")
        source_layout = QtWidgets.QFormLayout(self.source_group)
        self.left_selector = QtWidgets.QComboBox()
        self.right_selector = QtWidgets.QComboBox()
        self.refresh_runs_btn = QtWidgets.QPushButton("Refresh Database")
        
        source_layout.addRow("Left Panel:", self.left_selector)
        source_layout.addRow("Right Panel:", self.right_selector)
        source_layout.addRow("", self.refresh_runs_btn)
        self.sidebar_layout.addWidget(self.source_group)

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
        self.maze_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.maze_spinbox.setAlignment(QtCore.Qt.AlignCenter)
                    
        top_nav_layout.addWidget(self.maze_label)
        top_nav_layout.addSpacing(5)
        top_nav_layout.addWidget(self.maze_spinbox)
        top_nav_layout.addStretch()
        
        self.maze_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maze_slider.setRange(0, 999)
        
        self.maze_spinbox.setStyleSheet("QSpinBox { padding: 4px; border-radius: 6px; border: 1px solid #444; background-color: #1e1e1e; color: #ddd; }")
        self.maze_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #2a2a2a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00BFFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #00BFFF; border-radius: 3px; }
        """)
        self.maze_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #333; border-radius: 8px; margin-top: 10px; padding: 8px; } QGroupBox::title { left: 10px; padding: 0 3px; }")
        
        maze_nav_layout.addLayout(top_nav_layout)
        maze_nav_layout.addWidget(self.maze_slider)
        self.sidebar_layout.addWidget(self.maze_group)

        # --- Stats Group ---
        self.stats_group = QtWidgets.QGroupBox("Statistical Performance Profile")
        stats_layout = QtWidgets.QVBoxLayout(self.stats_group)
        self.run_test_btn = QtWidgets.QPushButton("Run Batch Test on Right Panel")
        self.run_test_btn.setMinimumHeight(40)
        self.run_test_btn.setStyleSheet("background-color: #28A745; color: white; font-weight: bold; border-radius: 5px;")
        
        self.stats_display = QtWidgets.QLabel("Select a dataset and run test...")
        self.stats_display.setStyleSheet("font-family: 'Courier New'; font-size: 10pt; color: #00FF00; background-color: #000; padding: 8px;")
        
        stats_layout.addWidget(self.run_test_btn)
        stats_layout.addWidget(self.stats_display)
        self.sidebar_layout.insertWidget(2, self.stats_group)

        # --- Micro-Inspector (Neuro-Probe) ---
        self.inspect_group = QtWidgets.QGroupBox("Neuro-Probe: Q-Values")
        inspect_layout = QtWidgets.QVBoxLayout(self.inspect_group)
        
        self.neuro_label = QtWidgets.QLabel("Click on any cell to probe its internal state.")
        self.neuro_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11pt; color: #00BFFF; background-color: #1e1e1e; padding: 10px; border-radius: 5px;")
        self.neuro_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        inspect_layout.addWidget(self.neuro_label)
        self.sidebar_layout.addWidget(self.inspect_group)

        # --- Probes Info Legend ---
        self.probes_help = QtWidgets.QGroupBox("Probes Guide")
        help_layout = QtWidgets.QVBoxLayout(self.probes_help)
        self.probes_help.setStyleSheet("QGroupBox { color: #FFD700; }")
        help_text = QtWidgets.QLabel(
            "<b>• CLICK:</b> Run Micro-Probe on cell.<br>"
            "<b>• YELLOW/BLUE BOX:</b> Aliased states.<br>"
            "<b>• CONFLICT:</b> Mathematical solvability.<br>"
            "<b>• ENTROPY:</b> Policy uncertainty (Ties).<br>"
            "<b>• PATH:</b> Visualizes Limit Cycles (Loops)."
        )
        help_text.setStyleSheet("font-size: 9pt; color: #BBB;")
        help_layout.addWidget(help_text)
        self.sidebar_layout.addWidget(self.probes_help)

        # --- Config/Metadata Inspector ---
        self.config_group = QtWidgets.QGroupBox("Right Panel Configuration (JSON)")
        config_layout = QtWidgets.QVBoxLayout(self.config_group)
        self.config_display = QtWidgets.QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: 'Courier New'; font-size: 10pt;")
        config_layout.addWidget(self.config_display)
        self.sidebar_layout.addWidget(self.config_group)

        self.sidebar_layout.addStretch()

        # ============================
        # VIEWER PANELS (RIGHT SIDE OF SPLITTER)
        # ============================
        self.viewer_widget = QtWidgets.QWidget()
        self.viewer_layout = QtWidgets.QHBoxLayout(self.viewer_widget)
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        tab_css = """
            QTabWidget::pane { border: 1px solid #333; top: -1px; background: #121212; }
            QTabBar::tab { background: #2a2a2a; color: #aaa; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #121212; color: #00BFFF; border: 1px solid #333; border-bottom: none; }
        """

        # --- LEFT TABS ---
        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.setStyleSheet(tab_css)
        self.view_left_policy = MazeView(title="Left: Policy (π)")
        self.view_left_values = MazeView(title="Left: Values (V)")
        self.view_left_entropy = MazeView(title="Left: Entropy (H)")
        self.left_tabs.addTab(self.view_left_policy, "Policy (π)")
        self.left_tabs.addTab(self.view_left_values, "Value (V)")
        self.left_tabs.addTab(self.view_left_entropy, "Entropy (H)")

        # --- RIGHT TABS ---
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.setStyleSheet(tab_css)
        self.view_right_policy = MazeView(title="Right: Policy (π)")
        self.view_right_values = MazeView(title="Right: Values (V)")
        self.view_right_entropy = MazeView(title="Right: Entropy (H)")
        self.right_tabs.addTab(self.view_right_policy, "Policy (π)")
        self.right_tabs.addTab(self.view_right_values, "Value (V)")
        self.right_tabs.addTab(self.view_right_entropy, "Entropy (H)")

        self.viewer_splitter.addWidget(self.left_tabs)
        self.viewer_splitter.addWidget(self.right_tabs)
        self.viewer_splitter.setSizes([800, 800])

        self.viewer_layout.addWidget(self.viewer_splitter)

        # Assemble Main Splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.viewer_widget)
        self.splitter.setSizes([300, 1100])
        self.inspector_layout.addWidget(self.splitter)

        self.workspaces.addWidget(self.inspector_page) # Index 0

        # --- WORKSPACE 2: ANALYTICS LAB ---
        self.analytics_page = QtWidgets.QLabel("Analytics Lab: Global Statistical Benchmarking\n(Coming Soon)")
        self.analytics_page.setStyleSheet("color: #555; font-size: 20pt;")
        self.analytics_page.setAlignment(QtCore.Qt.AlignCenter)
        self.workspaces.addWidget(self.analytics_page) # Index 1

    def connect_signals(self):
        """Connects UI elements to logic methods."""
        self.refresh_runs_btn.clicked.connect(self.refresh_runs)
        
        # New Dropdowns
        self.left_selector.currentIndexChanged.connect(self.on_left_selected)
        self.right_selector.currentIndexChanged.connect(self.on_right_selected)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_selected)
        
        # Sync Slider and Spinbox
        self.maze_slider.valueChanged.connect(self.maze_spinbox.setValue)
        self.maze_spinbox.valueChanged.connect(self.maze_slider.setValue)
        self.maze_slider.valueChanged.connect(self.update_display)

        self.run_test_btn.clicked.connect(self.run_batch_test)
        self.workspace_group.buttonClicked[int].connect(self.on_workspace_changed)

        # Connect ALL 6 views to the neuro probe
        for view in[self.view_left_policy, self.view_left_values, self.view_left_entropy,
                     self.view_right_policy, self.view_right_values, self.view_right_entropy]:
            view.cellClicked.connect(self.update_neuro_probe)
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_Right:
            self.maze_slider.setValue(self.maze_slider.value() + 1)
        elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_Left:
            self.maze_slider.setValue(self.maze_slider.value() - 1)
        else:
            super().keyPressEvent(event)

    def on_workspace_changed(self, index):
        self.workspaces.setCurrentIndex(index)

    # ============================
    # DATA & LOGIC METHODS
    # ============================

    def refresh_runs(self):
        runs = db_manager.get_all_runs()
        items = ["Oracle (Value Iteration)"] + runs
        
        for selector in [self.left_selector, self.right_selector]:
            selector.blockSignals(True)
            selector.clear()
            selector.addItems(items)
            selector.blockSignals(False)
            
        self.left_selector.setCurrentIndex(0)
        if len(items) > 1:
            self.right_selector.setCurrentIndex(1)
        self.on_left_selected()
        self.on_right_selected()

    def scan_datasets(self):
        self.dataset_selector.blockSignals(True)
        path = "data_jax"
        if os.path.exists(path):
            files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
            self.dataset_selector.addItems(files)
        self.dataset_selector.blockSignals(False)
        
        if self.dataset_selector.count() > 0:
            self.dataset_selector.setCurrentIndex(0)
            self.on_dataset_selected()

    def _load_selection(self, text_val):
        if text_val == "Oracle (Value Iteration)" or not text_val:
            return 'oracle', None, None, None, None # Added 5th None
            
        details = db_manager.get_run_details(text_val)
        if details and os.path.exists(details['path']):
            r_key = details['state_repr']
            q = jnp.array(np.load(details['path']))
            dec = core_logic.DECODERS.get(r_key, core_logic.decode_mdp)
            dec_jit = core_logic.STATE_MAP_FUNCS_JIT.get(r_key, core_logic.STATE_MAP_FUNCS_JIT['mdp'])
            dec_raw = core_logic.STATE_MAP_FUNCS_RAW.get(r_key, core_logic.STATE_MAP_FUNCS_RAW['mdp']) # NEW
            return 'agent', q, dec, dec_jit, dec_raw
        return 'oracle', None, None, None, None
    

    def on_left_selected(self):
        val = self.left_selector.currentText()
        self.left_type, self.left_q, self.left_decoder, self.left_decoder_jit, self.left_decoder_raw = self._load_selection(val)
        self._refresh_rollout_caches() # Cache!
        self.update_display()

    def on_right_selected(self):
        val = self.right_selector.currentText()
        self.right_type, self.right_q, self.right_decoder, self.right_decoder_jit, self.right_decoder_raw = self._load_selection(val)
        if val != "Oracle (Value Iteration)":
            details = db_manager.get_run_details(val)
            if details: self.config_display.setText(json.dumps(details['config'], indent=4))
        else:
            self.config_display.setText("Oracle (Value Iteration) loaded.")
        self._refresh_rollout_caches() # Cache!
        self.update_display()

    def on_dataset_selected(self):
        ds_name = self.dataset_selector.currentText()
        if not ds_name: return
        
        self.current_mazes = np.load(os.path.join("data_jax", ds_name))
        self.current_mazes_jax = jnp.array(self.current_mazes)
        
        vi_filename = ds_name.replace(".npy", "_VI_solved.npz")
        vi_path = os.path.join("data_jax", "value_iteration", vi_filename)
        
        if os.path.exists(vi_path):
            with np.load(vi_path) as data:
                self.current_vi_policies = data['policy']
                self.current_vi_values = data['values']
        else:
            self.current_vi_policies = None
            self.current_vi_values = None

        self._refresh_rollout_caches() 
        self.update_display()
    
    def _refresh_rollout_caches(self):
        """Silently pre-computes rollouts for all 1000 mazes to guarantee 0-lag sliders."""
        if self.current_mazes_jax is None: return

        # 1. Cache Left Agent
        if self.left_type == 'agent' and self.left_q is not None and self.left_decoder_raw is not None:
            res_l = core_logic.evaluate_dataset(self.left_q, self.current_mazes_jax, self.left_decoder_raw)
            self.left_rollout_cache = {'steps': np.array(res_l[2]), 'success': np.array(res_l[4])}
        else:
            self.left_rollout_cache = None

        # 2. Cache Right Agent
        if self.right_type == 'agent' and self.right_q is not None and self.right_decoder_raw is not None:
            res_r = core_logic.evaluate_dataset(self.right_q, self.current_mazes_jax, self.right_decoder_raw)
            self.right_rollout_cache = {'steps': np.array(res_r[2]), 'success': np.array(res_r[4])}
        else:
            self.right_rollout_cache = None

    def run_batch_test(self):
        # We run the batch test on whatever is loaded in the RIGHT panel
        if self.right_type != 'agent' or self.right_q is None or self.current_mazes_jax is None:
            self.stats_display.setText("Load an Agent in the Right Panel to test.")
            return
            
        self.run_test_btn.setText("Computing Rollouts...")
        QtWidgets.QApplication.processEvents()
        
        # Fetch the raw decoder required for JAX batch testing
        details = db_manager.get_run_details(self.right_selector.currentText())
        r_key = details['state_repr']
        raw_decoder = core_logic.STATE_MAP_FUNCS_RAW.get(r_key, core_logic.STATE_MAP_FUNCS_RAW['mdp'])

        res = core_logic.evaluate_dataset(self.right_q, self.current_mazes_jax, raw_decoder)
        
        steps_arr = np.array(res[2])
        colls_arr = np.array(res[3])
        goal_arr = np.array(res[4])
        
        total = len(goal_arr)
        success_count = np.sum(goal_arr)
        success_rate = (success_count / total) * 100
        timeout_rate = (np.sum(steps_arr >= 500) / total) * 100
        
        avg_steps = 0
        avg_colls = 0
        opt_gap = 1.0
        
        if success_count > 0:
            success_mask = (goal_arr == True)
            avg_steps = np.mean(steps_arr[success_mask])
            avg_colls = np.mean(colls_arr[success_mask])
            
            if self.current_vi_policies is not None:
                oracle_steps_all = np.abs(self.current_vi_values[:, 0, 0])
                avg_oracle_steps = np.mean(oracle_steps_all[success_mask])
                opt_gap = avg_steps / avg_oracle_steps

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
        self.run_test_btn.setText("Run Batch Test on Right Panel")

    def update_neuro_probe(self, r, c, state_id, aliased_count):
        # If toggled off
        if r == -1:
            self.neuro_label.setText("Click a cell to probe its internal state.")
            for view in[self.view_left_policy, self.view_left_values, self.view_left_entropy,
                         self.view_right_policy, self.view_right_values, self.view_right_entropy]:
                view.clear_highlights()
                view.clear_trajectory()
            return
        
        # --- BUG 1 FIX: SYNC THE STATE MAPS FOR HIGHLIGHTING ---
        idx = self.maze_slider.value()
        maze_jax = self.current_mazes_jax[idx]
        
        # Determine which decoder the clicked panel is using
        active_decoder = self.right_decoder_jit if self.right_type == 'agent' else self.left_decoder_jit
        if active_decoder is not None:
            shared_state_map = active_decoder(maze_jax)
            
            # Force ALL views to use the clicked panel's state map for the highlight
            self.view_left_policy.set_state_map(shared_state_map)
            self.view_left_values.set_state_map(shared_state_map)
            self.view_left_entropy.set_state_map(shared_state_map)
        # -------------------------------------------------------

        # Highlight ALL views
        blue_rgba =[0, 191, 255, 120]
        gold_rgba =[255, 215, 0, 120]
        L_color = blue_rgba if self.left_type == 'oracle' else gold_rgba
        R_color = blue_rgba if self.right_type == 'oracle' else gold_rgba

        self.view_left_policy.highlight_aliased_states(state_id, L_color)
        self.view_left_values.highlight_aliased_states(state_id, L_color)
        self.view_left_entropy.highlight_aliased_states(state_id, L_color)
        
        self.view_right_policy.highlight_aliased_states(state_id, R_color)
        self.view_right_values.highlight_aliased_states(state_id, R_color)
        self.view_right_entropy.highlight_aliased_states(state_id, R_color)

        # We probe the Right Agent by default. If Right is Oracle, probe Left.
        agent_q = self.right_q if self.right_q is not None else self.left_q
        agent_decoder = self.right_decoder if self.right_decoder is not None else self.left_decoder

        if agent_q is not None:
            q_vals = agent_q[state_id]
            entropy = core_logic.calculate_entropy(q_vals)
            
            idx = self.maze_slider.value()
            oracle_p = self.current_vi_policies[idx] if self.current_vi_policies is not None else None
            
            if oracle_p is not None and hasattr(self, 'right_decoder_jit') and self.right_decoder_jit is not None:
                state_map = self.right_decoder_jit(self.current_mazes_jax[idx])
                conflict = core_logic.calculate_conflict(state_id, state_map, oracle_p)
            else:
                conflict = 0.0
            
            text = (
                f"<b style='color:#FFD700;'>State ID:</b> {state_id}<br>"
                f"<b style='color:#FFD700;'>Aliased:</b> {aliased_count} locations<br>"
                f"<b style='color:#FFD700;'>Conflict:</b> {conflict:.1f}%<br>"
                f"<b style='color:#FFD700;'>Entropy:</b> {entropy:.3f}<br>"
                f"<hr style='border: 1px solid #444;'>"
                f"<b>[↑] Up   :</b> {q_vals[0]:>8.2f}<br>"
                f"<b>[↓] Down :</b> {q_vals[1]:>8.2f}<br>"
                f"<b>[←] Left :</b> {q_vals[2]:>8.2f}<br>"
                f"<b>[→] Right:</b> {q_vals[3]:>8.2f}<br>"
            )
            self.neuro_label.setText(text)

            # Draw Trajectory
            maze = self.current_mazes[idx]
            path = core_logic.compute_rollout(maze, (r, c), agent_q, agent_decoder)
            self.view_right_policy.draw_trajectory(path)
            self.view_right_values.draw_trajectory(path)
            self.view_left_policy.draw_trajectory(path)
            self.view_left_values.draw_trajectory(path)

    def _get_panel_data(self, panel_type, q_table, decoder_jit, idx, maze_jax):
        """Helper to extract Actions, Values, Entropy, and StateMap."""
        if panel_type == 'oracle':
            if self.current_vi_policies is None or self.current_vi_values is None: 
                return None, None, None, None
            actions = self.current_vi_policies[idx]
            values = self.current_vi_values[idx]
            entropy = np.zeros((16, 16))
            state_map = core_logic.STATE_MAP_FUNCS_JIT['mdp'](maze_jax)
            return actions, values, entropy, state_map
        else:
            if q_table is None or decoder_jit is None: 
                return None, None, None, None
            state_map = decoder_jit(maze_jax)
            q_vectors = q_table[state_map]
            actions = np.argmax(q_vectors, axis=-1)
            values = np.max(q_vectors, axis=-1)
            entropy = core_logic.calculate_entropy_grid(q_vectors)
            return actions, values, entropy, state_map

    def update_display(self):
        idx = self.maze_slider.value()
        if self.current_mazes is None: return
        
        maze = self.current_mazes[idx]
        maze_jax = self.current_mazes_jax[idx]

        for view in[self.view_left_policy, self.view_left_values, self.view_left_entropy,
                     self.view_right_policy, self.view_right_values, self.view_right_entropy]:
            view.set_maze(maze)

        L_act, L_val, L_ent, L_map = self._get_panel_data(self.left_type, self.left_q, self.left_decoder_jit, idx, maze_jax)
        R_act, R_val, R_ent, R_map = self._get_panel_data(self.right_type, self.right_q, self.right_decoder_jit, idx, maze_jax)

        # 1. Draw Left Panel (Green arrows)
        if L_act is not None:
            self.view_left_policy.set_state_map(L_map)
            self.view_left_values.set_state_map(L_map)
            self.view_left_entropy.set_state_map(L_map)
            
            self.view_left_policy.draw_policy_vectorized(maze, L_act, base_color='#28A745')
            self.view_left_values.set_heatmap(maze, L_val)
            self.view_left_entropy.set_heatmap(maze, L_ent)

        # 2. Draw Right Panel (Blue arrows, Red if diverges from Left)
        if R_act is not None:
            self.view_right_policy.set_state_map(R_map)
            self.view_right_values.set_state_map(R_map)
            self.view_right_entropy.set_state_map(R_map)
            
            # Left actions act as the truth for divergence checking
            self.view_right_policy.draw_policy_vectorized(maze, R_act, oracle_actions=L_act, base_color='#00BFFF')
            self.view_right_values.set_heatmap(maze, R_val)
            self.view_right_entropy.set_heatmap(maze, R_ent)
            
            # Scoreboard Logic
            # --- NEW FAST SCOREBOARD LOGIC ---
        oracle_steps = int(abs(self.current_vi_values[idx, 0, 0])) if self.current_vi_values is not None else 0

        # Title for LEFT Panel
        if self.left_type == 'oracle':
            self.view_left_policy.setTitle(f"Left Oracle: <span style='color:#00FF00;'>OPTIMAL</span> | Steps: {oracle_steps}", size='11pt')
        elif self.left_rollout_cache is not None:
            l_reached = self.left_rollout_cache['success'][idx]
            l_steps = self.left_rollout_cache['steps'][idx]
            l_status = "<span style='color:#00FF00;'>SUCCESS</span>" if l_reached else "<span style='color:#FF4500;'>FAILED</span>"
            l_metric = f"Steps: {l_steps} (Oracle: {oracle_steps})" if l_reached else "Trapped in Loop"
            self.view_left_policy.setTitle(f"Left Agent: {l_status} | {l_metric}", size='11pt')

        # Title for RIGHT Panel
        if self.right_type == 'oracle':
            self.current_maze_score = f"Right Oracle: <span style='color:#00FF00;'>OPTIMAL</span> | Steps: {oracle_steps}"
            self.view_right_policy.setTitle(self.current_maze_score, size='11pt')
        elif self.right_rollout_cache is not None:
            r_reached = self.right_rollout_cache['success'][idx]
            r_steps = self.right_rollout_cache['steps'][idx]
            r_status = "<span style='color:#00FF00;'>SUCCESS</span>" if r_reached else "<span style='color:#FF4500;'>FAILED</span>"
            r_metric = f"Steps: {r_steps} (Oracle: {oracle_steps})" if r_reached else "Trapped in Loop"
            self.current_maze_score = f"Right Agent: {r_status} | {r_metric}"
            self.view_right_policy.setTitle(self.current_maze_score, size='11pt')