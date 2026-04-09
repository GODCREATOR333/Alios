import numpy as np
import jax
import jax.numpy as jnp
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from viewer_2d import MazeView
import stochastic_models as models
import swarm_engine as engine

import db_manager

class SandboxView(QtWidgets.QWidget):
    mazeChanged = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        
        # --- STYLING (Matched to Deep Inspector) ---
        self.slider_style = """
            QSlider::groove:horizontal { height: 6px; background: #2a2a2a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00BFFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #00BFFF; border-radius: 3px; }
        """
        self.group_style = """
        QGroupBox { font-weight:bold; border:1px solid #333; border-radius:8px; margin-top:10px; padding:8px; }
        QGroupBox::title { left:10px; padding:0 3px; }
        """

        self.calc_timer = QtCore.QTimer()
        self.calc_timer.setSingleShot(True)
        self.calc_timer.timeout.connect(self._run_simulation_now)

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- SIDEBAR ---
        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFixedWidth(400)
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar_scroll.setStyleSheet("""
        QScrollBar:vertical { border:none; background:#121212; width:8px; }
        QScrollBar::handle:vertical { background:#555; border-radius:4px; }
        QScrollBar::handle:vertical:hover { background:#777; }
        """)
        
        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 15, 10)
        self.sidebar_layout.setSpacing(12)

        # ZONE A: STATISTICAL INFO
        self.stats_group = QtWidgets.QGroupBox("Statistical Analytics")
        stats_lay = QtWidgets.QVBoxLayout(self.stats_group)
        self.stats_label = QtWidgets.QLabel("Initializing Lab...")
        self.stats_label.setStyleSheet("""
            color: #CCCCCC;
            font-family: 'Courier New';
            font-size: 11pt;
            background: #0d0d0d;
            padding: 10px;
            border-radius: 6px;
            line-height: 140%;
            """)
        stats_lay.addWidget(self.stats_label)
        self.sidebar_layout.addWidget(self.stats_group)

        # ZONE B: NAVIGATION
        self.nav_group = QtWidgets.QGroupBox("Maze Navigator")
        self.nav_group.setStyleSheet(self.group_style)
        nav_lay = QtWidgets.QVBoxLayout(self.nav_group)
        self.ds_label = QtWidgets.QLabel("DS: None")
        self.ds_label.setStyleSheet("color: #777; font-size: 9pt; font-weight: bold;")
        nav_lay.addWidget(self.ds_label)

        spin_row = QtWidgets.QHBoxLayout()
        self.maze_spin = QtWidgets.QSpinBox()
        self.maze_spin.setRange(0, 999)
        self.maze_spin.setFixedWidth(60)
        self.maze_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.maze_spin.setAlignment(QtCore.Qt.AlignCenter)
        self.maze_spin.setStyleSheet("QSpinBox { padding:4px; border-radius:6px; border:1px solid #444; background:#1e1e1e; color:#ddd; }")
        self.maze_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maze_slider.setRange(0, 999)
        self.maze_slider.setStyleSheet(self.slider_style)
        spin_row.addWidget(QtWidgets.QLabel("Maze ID:"))
        spin_row.addWidget(self.maze_spin)
        nav_lay.addLayout(spin_row)
        nav_lay.addWidget(self.maze_slider)
        self.sidebar_layout.addWidget(self.nav_group)

        # ZONE C: SIMULATION SETTINGS
        self.settings_group = QtWidgets.QGroupBox("Engine Configuration")
        self.settings_group.setStyleSheet(self.group_style)
        set_lay = QtWidgets.QFormLayout(self.settings_group)
        
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems(["Geocentric (Goal Drift)", "Egocentric (Active Matter)", "Hybrid (GEO ↔ EGO)","RL Meta-Agent (Learned)"])
        self.lens_selector = QtWidgets.QComboBox()
        self.lens_selector.addItems(["Probability Cloud", "Ghost Swarm", "Representative Trajectory (Modes)", "Fluid Vector Field"])
        
        self.sim_ghosts = self._add_param_slider(set_lay, "Ghosts:", 10, 500, 100)
        self.sim_steps = self._add_param_slider(set_lay, "Max Steps:", 10, 500, 150)
        
        set_lay.addRow("Architecture:", self.model_selector)
        set_lay.addRow("View Lens:", self.lens_selector)
        self.sidebar_layout.addWidget(self.settings_group)

        # ZONE D: PHYSICS PARAMS
        self.l_card = QtWidgets.QGroupBox("Left Panel (Green)")
        l_lay = QtWidgets.QVBoxLayout(self.l_card)
        self.l_kappa = self._add_param_slider(l_lay, "Goal Drift (κ):", 0, 100, 25, "#6FCF97")
        self.l_dr = self._add_param_slider(l_lay, "Inertia (Dr):", 0, 100, 15, "#6FCF97")
        self.l_gamma = self._add_param_slider(l_lay, "Switching (γ):", 0, 100, 50, "#6FCF97")
        self.sidebar_layout.addWidget(self.l_card)

        self.r_card = QtWidgets.QGroupBox("Right Panel (Blue)")
        r_lay = QtWidgets.QVBoxLayout(self.r_card)
        self.r_kappa = self._add_param_slider(r_lay, "Goal Drift (κ):", 0, 100, 10, "#56CCF2")
        self.r_dr = self._add_param_slider(r_lay, "Inertia (Dr):", 0, 100, 50, "#56CCF2")
        self.r_gamma = self._add_param_slider(r_lay, "Switching (γ):", 0, 100, 20, "#56CCF2")
        self.sidebar_layout.addWidget(self.r_card)

        self.sidebar_layout.addStretch()      
        self.sidebar_scroll.setWidget(self.sidebar)

        # --- VIEWERS ---
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.view_left = MazeView(title="Mode A")
        self.view_right = MazeView(title="Mode B")
        self.viewer_splitter.addWidget(self.view_left)
        self.viewer_splitter.addWidget(self.view_right)
        self.viewer_splitter.setSizes([800, 800])

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.sidebar_scroll)
        self.splitter.addWidget(self.viewer_splitter)
        self.splitter.setSizes([320, 1080])
        self.main_layout.addWidget(self.splitter)

        # --- CONNECTIONS ---
        self.maze_slider.valueChanged.connect(self.maze_spin.setValue)
        self.maze_spin.valueChanged.connect(self.maze_slider.setValue)
        self.maze_slider.valueChanged.connect(lambda v: self.mazeChanged.emit(v))
        
        widgets = [self.l_kappa, self.l_dr, self.l_gamma, self.r_kappa, self.r_dr, self.r_gamma, 
                   self.model_selector, self.lens_selector, self.sim_ghosts, self.sim_steps, self.maze_slider]
        for w in widgets:
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(self._trigger_update)
            else: w.currentIndexChanged.connect(self._trigger_update)

        self.current_maze = None
        self.current_maze_jax = None
        self.rng_key = jax.random.PRNGKey(np.random.randint(1000))
        self.path_items = []

    def _add_param_slider(self, layout, label, min_v, max_v, start_v, color="#00BFFF"):
        lbl = QtWidgets.QLabel(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(start_v)
        
        css = f"""
            QSlider::groove:horizontal {{ height: 6px; background: #2a2a2a; border-radius: 3px; }}
            QSlider::handle:horizontal {{ background: {color}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: {color}; border-radius: 3px; }}
        """
        slider.setStyleSheet(css)
        if isinstance(layout, QtWidgets.QFormLayout):
            layout.addRow(lbl, slider)
        else:
            layout.addWidget(lbl)
            layout.addWidget(slider)
        return slider

    def set_dataset_name(self, name):
        self.ds_label.setText(f"DS: {name}")
    
    def set_maze(self, maze_np, idx=None, connectivity=None):
        self.current_maze = maze_np
        self.current_maze_jax = jnp.array(maze_np)
        self.current_connectivity = connectivity # <--- ADD THIS LINE
        if idx is not None:
            self.maze_slider.blockSignals(True)
            self.maze_slider.setValue(idx); self.maze_spin.setValue(idx)
            self.maze_slider.blockSignals(False)
        self._trigger_update()

    def _trigger_update(self):
        self.calc_timer.start(150)

    def _run_simulation_now(self):
        if self.current_maze_jax is None: return
        
        # 1. Cleanup Overlays
        for view in [self.view_left, self.view_right]:
            view.clear_visuals(keep_image=False)
            for item in self.path_items: 
                try: view.removeItem(item)
                except: pass
        self.path_items = []

        # 2. Physics Configuration
        m_txt = self.model_selector.currentText()
        if "RL Meta-Agent" in m_txt:
            # Look up our registered agent
            details = db_manager.get_run_details("META_RL_V1_p10")
            if details:
                # Load the brain from the path in the DB
                q_data = jnp.array(np.load(details['path']))
                policy = models.LearnedMetaPolicy(q_data)
            else:
                # Fallback if you haven't run the registration script yet
                self.stats_label.setText("<b style='color:red;'>ERROR:</b> META_RL_V1_p10 not found in DB.")
                return
        elif "Hybrid" in m_txt: policy = models.HybridPolicy()
        elif "Geocentric" in m_txt: policy = models.GeoPolicy()
        else: policy = models.EgoPolicy()
        
        l_params = {'kappa': self.l_kappa.value()/10.0, 'D_R': self.l_dr.value()/100.0, 'gamma': self.l_gamma.value()/100.0, 'goal_pos': jnp.array([15.0, 15.0])}
        r_params = {'kappa': self.r_kappa.value()/10.0, 'D_R': self.r_dr.value()/100.0, 'gamma': self.r_gamma.value()/100.0, 'goal_pos': jnp.array([15.0, 15.0])}

        ghosts = self.sim_ghosts.value()
        steps = self.sim_steps.value()

        # 3. Execution
        self.rng_key, k1, k2 = jax.random.split(self.rng_key, 3)
        
        l_res = engine.simulate_swarm(k1, self.current_maze_jax, policy, l_params, jnp.array([0, 0]), ghosts, steps)
        
        r_res = engine.simulate_swarm(k2, self.current_maze_jax, policy, r_params, jnp.array([0, 0]), ghosts, steps)

        # Unpack: (paths, mems, acts, occupancy)
        l_paths, l_mems, l_acts, l_occ = l_res
        r_paths, r_mems, r_acts, r_occ = r_res

        def analyze_meta_behavior(mems_jax):
            mems = np.array(mems_jax)
            
            # If the array is 2D (ghosts, steps), it's a simple agent (no modes)
            if mems.ndim < 3:
                return 0.0
            
            # If 3D, index 0 is the Mode (0: GEO, 1: EGO)
            ego_usage = np.mean(mems[:, :, 0] == 1) * 100
            return ego_usage
        
        l_ego_pct = analyze_meta_behavior(l_mems)
        r_ego_pct = analyze_meta_behavior(r_mems)

        # 4. Statistical Calculations
        def get_stats(paths_jax):
            paths = np.array(paths_jax)
            reached = np.all(paths[:, -1] == [15, 15], axis=-1)
            sr = np.mean(reached) * 100
            dist_to_goal = np.linalg.norm(paths[:, -1] - [15, 15], axis=-1)
            return sr, np.mean(dist_to_goal), np.var(dist_to_goal)


        lsr, l_m, l_v = get_stats(l_paths)
        rsr, r_m, r_v = get_stats(r_paths)

        self.stats_label.setText(
            f"<b style='color:#FFD700'>DYNAMICS ANALYTICS</b><br>"
            f"Ghosts: {ghosts} | Horizon: {steps}<br>"
            f"<hr style='border:1px solid #444'>"
            f"<b style='color:#28A745'>MODE A (LEFT)</b><br>"
            f"Success Rate : {lsr:.1f}%<br>"
            f"EGO Usage    : {l_ego_pct:.1f}%<br>"
            f"Mean Final G : {l_m:.2f}<br>"
            f"Variance     : {l_v:.2f}<br><br>"
            f"<b style='color:#00BFFF'>MODE B (RIGHT)</b><br>"
            f"Success Rate : {rsr:.1f}%<br>"
            f"EGO Usage    : {r_ego_pct:.1f}%<br>"
            f"Mean Final G : {r_m:.2f}<br>"
            f"Variance     : {r_v:.2f}"
        )

        # 5. Rendering
        lens = self.lens_selector.currentText()
        self.view_left.set_maze(self.current_maze)
        self.view_right.set_maze(self.current_maze)

        if lens == "Probability Cloud":
            def prepare_data(occ_jax, conn_np):
                # 1. Start with the swarm data (log smoothed)
                data = np.log1p(np.array(occ_jax, dtype=float) * 20.0)
                # 2. If oracle says a cell is unreachable, force it to -inf
                if conn_np is not None:
                    unreachable = (conn_np < -500) | (~np.isfinite(conn_np))
                    data[unreachable] = -np.inf
                return data

            l_display = prepare_data(l_occ, self.current_connectivity)
            r_display = prepare_data(r_occ, self.current_connectivity)

            # 3. FIX: Ensure is_sparse=True is passed here!
            self.view_left.set_heatmap(self.current_maze, l_display, cmap='viridis', is_sparse=True)
            self.view_right.set_heatmap(self.current_maze, r_display, cmap='magma', is_sparse=True)
        elif lens == "Ghost Swarm":
            for i in range(min(15, ghosts)):
                self._draw_path(self.view_left, l_paths[i], '#28A74544')
                self._draw_path(self.view_right, r_paths[i], '#00BFFF44')
        elif lens == "Representative Trajectory (Modes)":
            if "Hybrid" in m_txt:
                self._draw_hybrid_path(self.view_left, l_paths[0], l_mems[0, :, 0])
                self._draw_hybrid_path(self.view_right, r_paths[0], r_mems[0, :, 0])
            else:
                self._draw_path(self.view_left, l_paths[0], '#28A745')
                self._draw_path(self.view_right, r_paths[0], '#00BFFF')
        elif lens == "Fluid Vector Field":
            field_l, _ = engine.compute_vector_field(self.current_maze_jax, policy, l_params)
            field_r, _ = engine.compute_vector_field(self.current_maze_jax, policy, r_params)
            self.view_left.draw_policy(self.current_maze, jnp.argmax(field_l, axis=-1), color='#28A745')
            self.view_right.draw_policy(self.current_maze, jnp.argmax(field_r, axis=-1), color='#00BFFF')

        self.view_left.setTitle(f"Mode A │ κ={l_params['kappa']:.1f}", color='#28A745')
        self.view_right.setTitle(f"Mode B │ κ={r_params['kappa']:.1f}", color='#00BFFF')

    def _draw_path(self, view, path_jax, color_hex):
        path = np.array(path_jax)
        curve = pg.PlotCurveItem(path[:,1], path[:,0], pen=pg.mkPen(pg.mkColor(color_hex), width=1.5))
        view.addItem(curve)
        self.path_items.append(curve)

    def _draw_hybrid_path(self, view, path_arr, modes_arr):
        path, modes = np.array(path_arr), np.array(modes_arr)
        for i in range(len(path) - 1):
            color = '#00BFFF' if modes[i] == 0 else '#FF4500'
            line = pg.PlotCurveItem([path[i,1], path[i+1,1]], [path[i,0], path[i+1,0]], pen=pg.mkPen(color, width=2.5))
            view.addItem(line)
            self.path_items.append(line)
            if i > 0 and modes[i] != modes[i-1]:
                scat = pg.ScatterPlotItem(x=[path[i,1]], y=[path[i,0]], symbol='d', size=12, brush='#FFD700', pen=None)
                view.addItem(scat); self.path_items.append(scat)