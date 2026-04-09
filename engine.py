import os
import json
import numpy as np
import jax.numpy as jnp
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import jax

# Local Imports
from viewer_2d import MazeView
from analytics_view import AnalyticsDashboard
from sandbox_view import SandboxView
import db_manager
import core_logic
import state_sandbox  # Registers the state decoders

# =========================================================================
# BACKGROUND WORKER (Analytics Lab)
# =========================================================================
class ExperimentWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int) # (Current task, Total tasks)
    finished = QtCore.pyqtSignal(dict)

    def __init__(self, selected_runs, selected_datasets):
        super().__init__()
        self.selected_runs = selected_runs
        self.selected_datasets = selected_datasets

    def run(self):
        master_results = {}
        total_tasks = len(self.selected_runs) * len(self.selected_datasets)
        current_task = 0

        for ds_name in self.selected_datasets:
            ds_path = os.path.join("data_jax", ds_name)
            mazes_jax = jnp.array(np.load(ds_path))
            
            vi_filename = ds_name.replace(".npy", "_VI_solved.npz")
            vi_path = os.path.join("data_jax", "value_iteration", vi_filename)
            oracle_v_start = None
            if os.path.exists(vi_path):
                with np.load(vi_path) as vi_data:
                    oracle_v_start = np.abs(vi_data['values'][:, 0, 0])

            for run_id in self.selected_runs:
                details = db_manager.get_run_details(run_id)
                if not details: continue
                
                q_table = jnp.array(np.load(details['path']))
                r_key = details['state_repr']
                map_func = core_logic.STATE_MAP_FUNCS_RAW.get(r_key, core_logic.STATE_MAP_FUNCS_RAW['mdp'])

                res = core_logic.evaluate_dataset(q_table, mazes_jax, map_func)
                pr_scores = core_logic.calculate_localization_batch(q_table, mazes_jax, map_func)
                P_batch = core_logic.compute_transition_matrix_batch(q_table, mazes_jax, map_func)
                ipr_batch = core_logic.calculate_ipr_statistics(P_batch)
                avg_msd = float(core_logic.calculate_msd(res))

                try: 
                    parts = ds_name.split('_')
                    density = float(parts[1].replace('P', '')) / 1000.0
                    category = parts[-1].replace('.npy', '') 
                except: 
                    density = 0.0
                    category = "empty"

                master_results[(run_id, ds_name)] = {
                    'success': (np.sum(res[4]) / len(res[4])) * 100,
                    'density': density,
                    'category': category,
                    'pr': np.mean(pr_scores), 
                    'ipr': float(jnp.mean(ipr_batch)),      
                    'msd': avg_msd,      
                    'reached_mask': np.array(res[4]),
                    'steps': np.array(res[2]),
                    'oracle_steps': oracle_v_start
                }

                current_task += 1
                self.progress.emit(current_task, total_tasks)
        
        self.finished.emit(master_results)

# =========================================================================
# MAIN ENGINE
# =========================================================================
class AliosWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALIOS: Mechanistic Interpreter")
        self.resize(1400, 900)

        # 1. State Management (Zone 1)
        self._init_state()

        # 2. UI & Signals
        self.init_ui()
        self.connect_signals()

        # 3. Startup Sequences
        self.refresh_runs()
        self.scan_datasets()
        
        # 4. Load initial policies
        self.left_selector.setCurrentIndex(0)
        self.right_selector.setCurrentIndex(0)
        self.on_left_selected()
        self.on_right_selected()

    # =========================================================================
    # ZONE 1: STATE MANAGEMENT & DATA LOADING
    # =========================================================================
    def _init_state(self):
        """Initializes the 'Backpack' of data the engine carries."""
        self.dataset_name = None
        self.mazes_np = None
        self.mazes_jax = None
        
        self.oracle_v = None
        self.oracle_pi = None
        self.oracle_q_batch = None
        
        # Unified Panel Dictionary
        self.panels = {
            'Left':  {'type': 'oracle', 'q': None, 'dec_scalar': None, 'dec_raw': None, 'dec_jit': None, 'cache': None, 'state_repr': 'mdp'},
            'Right': {'type': 'oracle', 'q': None, 'dec_scalar': None, 'dec_raw': None, 'dec_jit': None, 'cache': None, 'state_repr': 'mdp'}
        }
        
        self.sort_map = np.arange(1000)
        self.current_maze_score = ""

    def load_dataset(self, ds_name):
        """Pure data logic. Loads the maze and its corresponding Oracles."""
        if not ds_name: return
        self.dataset_name = ds_name
        self.mazes_np = np.load(os.path.join("data_jax", ds_name))
        self.mazes_jax = jnp.array(self.mazes_np)
        
        vi_path = os.path.join("data_jax", "value_iteration", ds_name.replace(".npy", "_VI_solved.npz"))
        if os.path.exists(vi_path):
            with np.load(vi_path) as data:
                self.oracle_pi = data['policy']
                self.oracle_v = data['values']
        else:
            self.oracle_pi, self.oracle_v = None, None

        ql_path = os.path.join("data_jax", "q-learning-optimal", ds_name.replace(".npy", "_QL_expert.npz"))
        if os.path.exists(ql_path):
            with np.load(ql_path) as data:
                self.oracle_q_batch = data['q_table']
        else:
            self.oracle_q_batch = None

    def load_panel_agent(self, side, agent_name):
        """Loads an agent's brain and eyes into a specific panel."""
        if agent_name == "Oracle (Value Iteration)" or not agent_name:
            self.panels[side].update({'type': 'oracle', 'q': None, 'dec_scalar': None, 'dec_raw': None, 'dec_jit': None, 'state_repr': 'mdp'})
            return

        details = db_manager.get_run_details(agent_name)
        if details and os.path.exists(details['path']):
            r_key = details['state_repr']
            self.panels[side].update({
                'type': 'agent',
                'q': jnp.array(np.load(details['path'])),
                'state_repr': r_key,
                'dec_scalar': core_logic.DECODERS.get(r_key, core_logic.DECODERS['mdp']),
                'dec_raw': core_logic.STATE_MAP_FUNCS_RAW.get(r_key, core_logic.STATE_MAP_FUNCS_RAW['mdp']),
                'dec_jit': core_logic.STATE_MAP_FUNCS_JIT.get(r_key, core_logic.STATE_MAP_FUNCS_JIT['mdp'])
            })
        else:
            self.panels[side].update({'type': 'oracle', 'q': None, 'dec_scalar': None, 'dec_raw': None, 'dec_jit': None, 'state_repr': 'mdp'})

    # =========================================================================
    # ZONE 2: UI EVENT HANDLERS & THE LAZY ROUTER
    # =========================================================================
    def on_dataset_selected(self):
        ds_name = self.dataset_selector.currentText()
        if not ds_name: return
        
        self.load_dataset(ds_name)
        
        num_mazes = len(self.mazes_np) if self.mazes_np is not None else 0
        self.maze_slider.blockSignals(True)
        self.maze_spinbox.blockSignals(True)
        self.maze_slider.setRange(0, num_mazes - 1)
        self.maze_spinbox.setRange(0, num_mazes - 1)
        self.maze_slider.blockSignals(False)
        self.maze_spinbox.blockSignals(False)

        if hasattr(self, 'sandbox_page'):
            self.sandbox_page.set_dataset_name(ds_name)

        self._refresh_rollout_caches()
        self.route_display_update()

    def on_left_selected(self):
        self.load_panel_agent('Left', self.left_selector.currentText())
        self._refresh_rollout_caches()
        self.route_display_update()

    def on_right_selected(self):
        val = self.right_selector.currentText()
        self.load_panel_agent('Right', val)
        
        if val != "Oracle (Value Iteration)":
            details = db_manager.get_run_details(val)
            if details: 
                self.config_display.setText(json.dumps(details['config'], indent=4))
        else:
            self.config_display.setText("Oracle (Value Iteration) loaded.")
            
        self._refresh_rollout_caches()
        self.route_display_update()

    def on_workspace_changed(self, index):
        self.workspaces.setCurrentIndex(index)
        self.route_display_update()

    def route_display_update(self):
        """THE TRAFFIC CONTROLLER: Only paints the active tab."""
        if self.mazes_np is None: return
        current_tab = self.workspaces.currentIndex()
        if current_tab == 0:
            self.update_inspector_display()
        elif current_tab == 2:
            self.sync_sandbox()

    def sync_sandbox(self):
        if hasattr(self, 'sandbox_page') and self.mazes_np is not None:
            idx = self.get_real_maze_idx()
            self.sandbox_page.set_maze(self.mazes_np[idx], idx=idx)

    # =========================================================================
    # ZONE 3: BACKGROUND PHYSICS & CACHING
    # =========================================================================
    def _refresh_rollout_caches(self):
        if self.mazes_jax is None: return

        for side in ['Left', 'Right']:
            p = self.panels[side]
            if p['type'] == 'agent' and p['q'] is not None and p['dec_raw'] is not None:
                res = core_logic.evaluate_dataset(p['q'], self.mazes_jax, p['dec_raw'])
                p['cache'] = {'steps': np.array(res[2]), 'success': np.array(res[4])}
            else:
                p['cache'] = None

        r_cache = self.panels['Right']['cache']
        if r_cache is not None and self.oracle_v is not None:
            agent_steps = r_cache['steps']
            oracle_steps = np.abs(self.oracle_v[:, 0, 0])
            success_mask = r_cache['success']
            gaps = np.where(success_mask, agent_steps / (oracle_steps + 1e-6), 999.0)
            self.sort_map = np.argsort(gaps)[::-1]
        else:
            self.sort_map = np.arange(len(self.mazes_np) if self.mazes_np is not None else 1000)

    # =========================================================================
    # ZONE 4: THE DEEP INSPECTOR (UI DRAWING)
    # =========================================================================
    def get_real_maze_idx(self):
        slider_val = self.maze_slider.value()
        if hasattr(self, 'sort_checkbox') and self.sort_checkbox.isChecked():
            return int(self.sort_map[slider_val])
        return slider_val

    def update_inspector_display(self):
        if self.mazes_np is None: return
        
        idx = self.get_real_maze_idx()
        slider_val = self.maze_slider.value()
        
        if self.sort_checkbox.isChecked():
            self.maze_label.setText(f"Rank: {slider_val} (Maze #{idx})")
        else:
            self.maze_label.setText(f"Maze Index: {idx}")
            
        maze = self.mazes_np[idx]
        maze_jax = self.mazes_jax[idx]
        oracle_steps = int(abs(self.oracle_v[idx, 0, 0])) if self.oracle_v is not None else 0

        panel_configs = [
            ('Left', '#28A745', [self.view_left_policy, self.view_left_values, self.view_left_entropy]),
            ('Right', '#00BFFF', [self.view_right_policy, self.view_right_values, self.view_right_entropy])
        ]

        l_actions, _, _, _ = self._get_panel_data('Left', idx, maze_jax)

        for name, color, views in panel_configs:
            for v in views:
                v.set_maze(maze)
                v.clear_visuals(keep_image=True)

            act, val, ent, smap = self._get_panel_data(name, idx, maze_jax)
            cache = self.panels[name]['cache']
            p_type = self.panels[name]['type']

            if act is not None:
                comparison_baseline = l_actions if name == 'Right' else None
                views[0].draw_policy(maze, act, comparison_grid=comparison_baseline, color=color)
                views[1].set_heatmap(maze, val)
                views[2].set_heatmap(maze, ent)

            if p_type == 'oracle':
                title = f"{name} Oracle: <span style='color:#00FF00;'>OPTIMAL</span> | Steps: {oracle_steps}"
            elif cache is not None:
                reached = cache['success'][idx]
                steps = cache['steps'][idx]
                status = "<span style='color:#00FF00;'>SUCCESS</span>" if reached else "<span style='color:#FF4500;'>FAILED</span>"
                metric = f"Steps: {steps} (Oracle: {oracle_steps})" if reached else "Trapped in Loop"
                title = f"{name} Agent: {status} | {metric}"
            else:
                title = f"{name} Panel"
            
            views[0].setTitle(title, size='11pt')
            if name == 'Right':
                self.current_maze_score = title

    def _get_panel_data(self, side, idx, maze_jax):
        p = self.panels[side]
        if p['type'] == 'oracle':
            if self.oracle_pi is None or self.oracle_v is None: 
                return None, None, None, None
            
            actions = self.oracle_pi[idx]
            
            # THE FIX: Filter out extreme penalties (<-500) so they become -inf
            # This fixes the yellow squishing and makes unreachable cells Red!
            values = np.array(self.oracle_v[idx], dtype=float)
            values[values < -500] = -np.inf
            
            entropy = np.zeros((16, 16))
            state_map = core_logic.STATE_MAP_FUNCS_JIT['mdp'](maze_jax)
            return actions, values, entropy, state_map
        else:
            if p['q'] is None or p['dec_jit'] is None: 
                return None, None, None, None
            
            state_map = p['dec_jit'](maze_jax)
            q_vectors = p['q'][state_map]
            actions = np.argmax(q_vectors, axis=-1)
            
            # Apply the same penalty filter for Q-tables
            values = np.array(np.max(q_vectors, axis=-1), dtype=float)
            values[values < -500] = -np.inf
            
            entropy = core_logic.calculate_entropy_grid(q_vectors)
            return actions, values, entropy, state_map

    def update_neuro_probe(self, r, c, state_id_argument, count_argument):
        if r == -1:
            self.neuro_label.setText("Click a cell to probe its internal state.")
            self.inspect_group.setTitle("Neuro-Probe")
            for view in [self.view_left_policy, self.view_left_values, self.view_left_entropy,
                         self.view_right_policy, self.view_right_values, self.view_right_entropy]:
                view.clear_probe() 
            return

        idx = self.get_real_maze_idx()
        maze_jax = self.mazes_jax[idx]
        maze_numpy = self.mazes_np[idx]

        active_p = self.panels['Right'] if self.panels['Right']['q'] is not None else self.panels['Left']
        
        if active_p['type'] == 'agent' and active_p['q'] is not None:
            active_jit = active_p['dec_jit']
            active_scalar = active_p['dec_scalar']
            active_q = active_p['q']
        elif self.oracle_q_batch is not None:
            active_jit = core_logic.STATE_MAP_FUNCS_JIT['mdp']
            active_scalar = core_logic.DECODERS['mdp']
            active_q = self.oracle_q_batch[idx]
        else:
            active_jit = core_logic.STATE_MAP_FUNCS_JIT['mdp']
            active_scalar = core_logic.DECODERS['mdp']
            active_q = None

        if active_jit is not None:
            state_id = int(active_scalar(maze_numpy, r, c))
            shared_map_jax = active_jit(maze_jax)
            shared_map_np = np.array(shared_map_jax)
            aliased_count = int(np.sum(shared_map_np == state_id))
            
            for v in [self.view_left_policy, self.view_left_values, self.view_left_entropy]:
                v.highlight_aliased_states(state_id, shared_map_np, maze_numpy, [0, 191, 255, 120])
            for v in [self.view_right_policy, self.view_right_values, self.view_right_entropy]:
                v.highlight_aliased_states(state_id, shared_map_np, maze_numpy, [255, 215, 0, 120])

            if active_q is not None:
                q_vals = active_q[state_id]
                entropy = core_logic.calculate_entropy(q_vals)
                conflict = 0.0
                if self.oracle_pi is not None:
                    oracle_p = self.oracle_pi[idx]
                    conflict = core_logic.calculate_conflict(state_id, shared_map_np, oracle_p)
                
                color_id = "#FFD700" 
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

                path = core_logic.compute_rollout(maze_numpy, (r, c), active_q, active_scalar)
                self.view_left_policy.draw_trajectory(path)
                self.view_right_policy.draw_trajectory(path)
            else:
                self.neuro_label.setText("Probing Oracle (MDP mode).<br>No Q-values available.")

    # =========================================================================
    # UI SETUP & ANALYTICS LAB METHODS
    # =========================================================================
    def init_ui(self):
        """Builds the layout and widgets."""
        self.main_container = QtWidgets.QVBoxLayout(self)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        self.main_container.setSpacing(0)

        # Header
        self.header = QtWidgets.QWidget()
        self.header.setMinimumHeight(55)
        self.header.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 5, 10, 0)
        
        btn_style = """
            QPushButton { background-color: transparent; color: #888; font-weight: bold; padding: 10px 20px; border: None; font-size: 10pt; }
            QPushButton:checked { color: #00BFFF; border-bottom: 2px solid #00BFFF; }
            QPushButton:hover { color: #DDD; }
        """
        self.workspace_group = QtWidgets.QButtonGroup(self)
        
        self.btn_inspector = QtWidgets.QPushButton("DEEP INSPECTOR")
        self.btn_analytics = QtWidgets.QPushButton("ANALYTICS LAB")
        self.btn_sandbox = QtWidgets.QPushButton("STOCHASTIC SANDBOX")
        
        for i, btn in enumerate([self.btn_inspector, self.btn_analytics, self.btn_sandbox]):
            btn.setStyleSheet(btn_style)
            btn.setCheckable(True)
            header_layout.addWidget(btn)
            self.workspace_group.addButton(btn, i)
        
        self.btn_inspector.setChecked(True)
        header_layout.addStretch()
        self.main_container.addWidget(self.header)

        self.workspaces = QtWidgets.QStackedWidget()
        self.main_container.addWidget(self.workspaces)

        # --- WORKSPACE 0: DEEP INSPECTOR ---
        self.inspector_page = QtWidgets.QWidget()
        self.inspector_layout = QtWidgets.QHBoxLayout(self.inspector_page)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True) 
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar_scroll.setMinimumWidth(320)
        
        self.sidebar_scroll.setStyleSheet("""
            QScrollBar:vertical { border: none; background: #121212; width: 8px; margin: 0px 0px 0px 0px; }
            QScrollBar::handle:vertical { background: #555; min-height: 30px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #777; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0px; }
        """)

        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 15, 10)
        self.sidebar_layout.setSpacing(12)

        self.source_group = QtWidgets.QGroupBox("Panel Sources")
        source_layout = QtWidgets.QFormLayout(self.source_group)
        self.left_selector = QtWidgets.QComboBox()
        self.right_selector = QtWidgets.QComboBox()
        self.refresh_runs_btn = QtWidgets.QPushButton("Refresh Database")
        source_layout.addRow("Left Panel:", self.left_selector)
        source_layout.addRow("Right Panel:", self.right_selector)
        source_layout.addRow("", self.refresh_runs_btn)
        self.sidebar_layout.addWidget(self.source_group)

        self.data_group = QtWidgets.QGroupBox("Test Dataset")
        data_layout = QtWidgets.QVBoxLayout(self.data_group)
        self.dataset_selector = QtWidgets.QComboBox()
        data_layout.addWidget(self.dataset_selector)
        self.sidebar_layout.addWidget(self.data_group)

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
        self.sort_checkbox = QtWidgets.QCheckBox("Sort by Failure Severity")
        self.sort_checkbox.setStyleSheet("color: #FF4500; font-weight: bold;")
        maze_nav_layout.addWidget(self.sort_checkbox)
        
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

        self.inspect_group = QtWidgets.QGroupBox("Neuro-Probe: Q-Values")
        inspect_layout = QtWidgets.QVBoxLayout(self.inspect_group)
        self.neuro_label = QtWidgets.QLabel("Click on any cell to probe its internal state.")
        self.neuro_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 11pt; color: #00BFFF; background-color: #1e1e1e; padding: 10px; border-radius: 5px;")
        self.neuro_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        inspect_layout.addWidget(self.neuro_label)
        self.sidebar_layout.addWidget(self.inspect_group)

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

        self.config_group = QtWidgets.QGroupBox("Right Panel Configuration (JSON)")
        config_layout = QtWidgets.QVBoxLayout(self.config_group)
        self.config_display = QtWidgets.QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: 'Courier New'; font-size: 10pt;")
        config_layout.addWidget(self.config_display)
        self.sidebar_layout.addWidget(self.config_group)

        self.sidebar_layout.addStretch()
        self.sidebar_scroll.setWidget(self.sidebar)

        self.viewer_widget = QtWidgets.QWidget()
        self.viewer_layout = QtWidgets.QHBoxLayout(self.viewer_widget)
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        tab_css = """
            QTabWidget::pane { border: 1px solid #333; top: -1px; background: #121212; }
            QTabBar::tab { background: #2a2a2a; color: #aaa; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #121212; color: #00BFFF; border: 1px solid #333; border-bottom: none; }
        """

        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.setStyleSheet(tab_css)
        self.view_left_policy = MazeView(title="Left: Policy (π)")
        self.view_left_values = MazeView(title="Left: Values (V)")
        self.view_left_entropy = MazeView(title="Left: Entropy (H)")
        self.left_tabs.addTab(self.view_left_policy, "Policy (π)")
        self.left_tabs.addTab(self.view_left_values, "Value (V)")
        self.left_tabs.addTab(self.view_left_entropy, "Entropy (H)")

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

        self.splitter.addWidget(self.sidebar_scroll)
        self.splitter.addWidget(self.viewer_widget)
        self.splitter.setSizes([320, 1080])
        self.inspector_layout.addWidget(self.splitter)
        self.workspaces.addWidget(self.inspector_page) # Index 0

        # --- WORKSPACE 1: ANALYTICS LAB ---
        self.analytics_page = QtWidgets.QWidget()
        analytics_main_layout = QtWidgets.QHBoxLayout(self.analytics_page)
        self.analytics_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.lab_sidebar = QtWidgets.QWidget()
        self.lab_sidebar.setMinimumWidth(300)
        lab_sidebar_layout = QtWidgets.QVBoxLayout(self.lab_sidebar)

        self.agent_list_group = QtWidgets.QGroupBox("Select Agents to Benchmark")
        agent_list_layout = QtWidgets.QVBoxLayout(self.agent_list_group)
        self.agent_list_widget = QtWidgets.QListWidget()
        self.agent_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        agent_list_layout.addWidget(self.agent_list_widget)
        lab_sidebar_layout.addWidget(self.agent_list_group)

        self.lab_dataset_group = QtWidgets.QGroupBox("Select Datasets to Test")
        lab_dataset_layout = QtWidgets.QVBoxLayout(self.lab_dataset_group)
        self.dataset_list_widget = QtWidgets.QListWidget()
        self.dataset_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lab_dataset_layout.addWidget(self.dataset_list_widget)
        lab_sidebar_layout.addWidget(self.lab_dataset_group)

        self.lab_lens_group = QtWidgets.QGroupBox("Scientific Lenses")
        lab_lens_layout = QtWidgets.QVBoxLayout(self.lab_lens_group)
        self.lens_list_widget = QtWidgets.QListWidget()
        
        for lens_name in['Success Rate vs Density', 'Optimality Scatter', 'Efficiency Distribution','Value Localization (Anderson)','Anderson Localization (IPR)','Mean Squared Displacement']:
            item = QtWidgets.QListWidgetItem(lens_name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked) 
            self.lens_list_widget.addItem(item)
            
        lab_lens_layout.addWidget(self.lens_list_widget)
        lab_sidebar_layout.addWidget(self.lab_lens_group)

        self.btn_run_benchmarks = QtWidgets.QPushButton("RUN AGENT COMPARISON")
        self.btn_run_benchmarks.setMinimumHeight(50)
        self.btn_run_benchmarks.setStyleSheet("background-color: #00BFFF; color: black; font-weight: bold; border-radius: 5px;")
        lab_sidebar_layout.addWidget(self.btn_run_benchmarks)
        lab_sidebar_layout.addStretch()

        self.plot_container = AnalyticsDashboard() 

        self.analytics_splitter.addWidget(self.lab_sidebar)
        self.analytics_splitter.addWidget(self.plot_container)
        self.analytics_splitter.setSizes([300, 1100])
        analytics_main_layout.addWidget(self.analytics_splitter)
        self.workspaces.addWidget(self.analytics_page) # Index 1

        # --- WORKSPACE 2: STOCHASTIC SANDBOX ---
        self.sandbox_page = SandboxView()
        self.workspaces.addWidget(self.sandbox_page) # Index 2

    def connect_signals(self):
        """Connects UI elements to logic methods."""
        self.refresh_runs_btn.clicked.connect(self.refresh_runs)
        
        self.left_selector.currentIndexChanged.connect(self.on_left_selected)
        self.right_selector.currentIndexChanged.connect(self.on_right_selected)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_selected)
        
        self.maze_slider.valueChanged.connect(self.maze_spinbox.setValue)
        self.maze_spinbox.valueChanged.connect(self.maze_slider.setValue)
        
        # THE FIX: Point the slider directly to the router
        self.maze_slider.valueChanged.connect(self.route_display_update)
        
        # Bridge the Sandbox back to the main slider
        self.sandbox_page.mazeChanged.connect(self.maze_slider.setValue)

        self.run_test_btn.clicked.connect(self.run_batch_test)
        self.workspace_group.buttonClicked[int].connect(self.on_workspace_changed)
        self.sort_checkbox.stateChanged.connect(self.route_display_update)

        for view in[self.view_left_policy, self.view_left_values, self.view_left_entropy,
                     self.view_right_policy, self.view_right_values, self.view_right_entropy]:
            view.cellClicked.connect(self.update_neuro_probe)
        
        self.btn_run_benchmarks.clicked.connect(self.execute_benchmarks)
        self.lens_list_widget.itemChanged.connect(self.on_lens_toggled)
        self.plot_container.mazeSelected.connect(self.jump_to_maze)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_Right:
            self.maze_slider.setValue(self.maze_slider.value() + 1)
        elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_Left:
            self.maze_slider.setValue(self.maze_slider.value() - 1)
        else:
            super().keyPressEvent(event)

    def on_lens_toggled(self, item):
        active_lenses =[self.lens_list_widget.item(i).text() 
                         for i in range(self.lens_list_widget.count()) 
                         if self.lens_list_widget.item(i).checkState() == QtCore.Qt.Checked]
        self.plot_container.update_active_lenses(active_lenses)

    def jump_to_maze(self, dataset_name, maze_idx):
        self.sort_checkbox.setChecked(False)
        found_idx = self.dataset_selector.findText(dataset_name)
        if found_idx >= 0:
            self.dataset_selector.setCurrentIndex(found_idx)
        self.maze_slider.setValue(int(maze_idx))
        self.btn_inspector.setChecked(True)
        self.on_workspace_changed(0)

    def refresh_runs(self):
        runs = db_manager.get_all_runs()
        items = ["Oracle (Value Iteration)"] + runs
        for selector in [self.left_selector, self.right_selector]:
            selector.blockSignals(True)
            selector.clear()
            selector.addItems(items)
            selector.blockSignals(False)

        self.agent_list_widget.clear()
        for run_id in runs:
            item = QtWidgets.QListWidgetItem(run_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.agent_list_widget.addItem(item)

    def scan_datasets(self):
        path = "data_jax"
        if os.path.exists(path):
            files = sorted([f for f in os.listdir(path) if f.endswith('.npy') and 'train' not in f])
            self.dataset_selector.blockSignals(True)
            self.dataset_selector.clear()
            self.dataset_selector.addItems(files)
            self.dataset_selector.blockSignals(False)
            
            self.dataset_list_widget.blockSignals(True)
            self.dataset_list_widget.clear()
            for f in files:
                item = QtWidgets.QListWidgetItem(f)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.dataset_list_widget.addItem(item)
            self.dataset_list_widget.blockSignals(False)
        
        if self.dataset_selector.count() > 0:
            self.dataset_selector.setCurrentIndex(0)
            self.on_dataset_selected()

    def run_batch_test(self):
        r_q = self.panels['Right']['q']
        r_dec = self.panels['Right']['dec_raw']
        if r_q is None or r_dec is None or self.mazes_jax is None:
            return
            
        self.run_test_btn.setText("Computing Rollouts...")
        QtWidgets.QApplication.processEvents()

        res = core_logic.evaluate_dataset(r_q, self.mazes_jax, r_dec)
        
        steps_arr = np.array(res[2])
        colls_arr = np.array(res[3])
        goal_arr = np.array(res[4])
        
        total = len(goal_arr)
        success_count = np.sum(goal_arr)
        success_rate = (success_count / total) * 100
        timeout_rate = (np.sum(steps_arr >= 500) / total) * 100
        global_avg_colls = np.mean(colls_arr) 
        
        avg_steps_success = 0
        opt_gap = 1.0

        safety_scores = jax.vmap(core_logic.calculate_policy_safety, in_axes=(None, 0, None))(
            r_q, self.mazes_jax, r_dec)
        avg_field_unsafe = np.mean(safety_scores)

        if success_count > 0:
            success_mask = (goal_arr == True)
            avg_steps_success = np.mean(steps_arr[success_mask])
            
            if self.oracle_pi is not None:
                oracle_steps_all = np.abs(self.oracle_v[:, 0, 0])
                avg_oracle = np.mean(oracle_steps_all[success_mask])
                opt_gap = avg_steps_success / (avg_oracle + 1e-6)

        self.stats_display.setText(
            f"<b>RESULT SUMMARY:</b><br>"
            f"-------------------------<br>"
            f"Success Rate : {success_rate:>6.1f}%<br>"
            f"Timeout Rate : {timeout_rate:>6.1f}%<br>"
            f"<b style='color: #FF4500;'>Policy Unsafe: {avg_field_unsafe:>6.1f}% (global)</b>"
            f"<b>Avg Collide  : {global_avg_colls:>6.1f}</b><br>"
            f"Avg Steps(S) : {avg_steps_success:>6.1f}<br>"
            f"-------------------------<br>"
            f"<b style='color: #FFD700;'>Optimality Gap: {opt_gap:.2f}x</b>"
        )
        self.run_test_btn.setText("Run Batch Test on Right Panel")

    def execute_benchmarks(self):
        selected_runs = [self.agent_list_widget.item(i).text() 
                         for i in range(self.agent_list_widget.count()) 
                         if self.agent_list_widget.item(i).checkState() == QtCore.Qt.Checked]
        
        selected_datasets = [self.dataset_list_widget.item(i).text() 
                             for i in range(self.dataset_list_widget.count()) 
                             if self.dataset_list_widget.item(i).checkState() == QtCore.Qt.Checked]

        if not selected_runs or not selected_datasets:
            self.stats_display.setText("<b style='color:red;'>Error:</b> Select agents and datasets.")
            return

        self.btn_run_benchmarks.setEnabled(False)
        self.btn_run_benchmarks.setText("Starting Analysis...")

        self.worker = ExperimentWorker(selected_runs, selected_datasets)
        self.worker.progress.connect(self._on_benchmark_progress)
        self.worker.finished.connect(self._on_benchmark_finished)
        self.worker.start()

    def _on_benchmark_progress(self, current, total):
        self.btn_run_benchmarks.setText(f"Processing: {current} / {total} Tasks")

    def _on_benchmark_finished(self, results):
        selected_runs = [self.agent_list_widget.item(i).text() for i in range(self.agent_list_widget.count()) if self.agent_list_widget.item(i).checkState() == QtCore.Qt.Checked]
        selected_datasets = [self.dataset_list_widget.item(i).text() for i in range(self.dataset_list_widget.count()) if self.dataset_list_widget.item(i).checkState() == QtCore.Qt.Checked]

        self.plot_container.set_data_cache(results, selected_runs, selected_datasets)
        self.on_lens_toggled(None)
        
        self.btn_run_benchmarks.setEnabled(True)
        self.btn_run_benchmarks.setText("RUN AGENT COMPARISON")
        self.stats_display.setText("<b style='color:#00FF00;'>Experiment Complete.</b>")