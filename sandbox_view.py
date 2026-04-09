import numpy as np
import jax
import jax.numpy as jnp
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from viewer_2d import MazeView
import stochastic_models as models
import swarm_engine as engine

class SandboxView(QtWidgets.QWidget):
    mazeChanged = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        
        # 1. Styles & Timers
        self.slider_style = """
            QSlider::groove:horizontal { height: 6px; background: #2a2a2a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00BFFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #00BFFF; border-radius: 3px; }
        """
        self.calc_timer = QtCore.QTimer()
        self.calc_timer.setSingleShot(True)
        self.calc_timer.timeout.connect(self._run_simulation_now)

        # 2. Main Layout
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 3. SIDEBAR
        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFixedWidth(260) # Slimmer width
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar_scroll.setStyleSheet("background-color: #121212; border-right: 1px solid #333;")
        
        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 15, 10, 15)
        self.sidebar_layout.setSpacing(12)

        def create_group(title):
            grp = QtWidgets.QGroupBox(title)
            grp.setStyleSheet("""
                QGroupBox { font-weight: bold; color: #00BFFF; border: 1px solid #333; 
                            border-radius: 6px; margin-top: 12px; padding: 10px 5px 5px 5px; }
                QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            """)
            return grp

        # --- NAVIGATOR ---
        self.nav_group = create_group("Navigator")
        nav_lay = QtWidgets.QVBoxLayout(self.nav_group)
        self.ds_label = QtWidgets.QLabel("DS: None")
        self.ds_label.setStyleSheet("color: #777; font-size: 8pt;")
        
        spin_row = QtWidgets.QHBoxLayout()
        self.maze_spin = QtWidgets.QSpinBox()
        self.maze_spin.setRange(0, 999)
        self.maze_spin.setFixedWidth(50)
        self.maze_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.maze_spin.setStyleSheet("background: #1e1e1e; color: #00BFFF; border: 1px solid #444;")
        
        self.maze_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maze_slider.setRange(0, 999)
        self.maze_slider.setStyleSheet(self.slider_style)
        
        spin_row.addWidget(QtWidgets.QLabel("ID:"))
        spin_row.addWidget(self.maze_spin)
        spin_row.addStretch()
        
        nav_lay.addWidget(self.ds_label)
        nav_lay.addLayout(spin_row)
        nav_lay.addWidget(self.maze_slider)
        self.sidebar_layout.addWidget(self.nav_group)

        # --- SETTINGS ---
        self.settings_group = create_group("Model & Lens")
        set_lay = QtWidgets.QFormLayout(self.settings_group)
        
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems(["Geocentric (Goal Drift)", "Egocentric (Active Matter)"])
        
        self.lens_selector = QtWidgets.QComboBox()
        self.lens_selector.addItems(["Probability Cloud", "Ghost Swarm", "Fluid Vector Field"])
        
        self.model_selector.setFixedWidth(140)
        self.lens_selector.setFixedWidth(140)

        set_lay.addRow("Model:", self.model_selector)
        set_lay.addRow("Lens:", self.lens_selector)
        self.sidebar_layout.addWidget(self.settings_group)

        # --- PARAMS ---
        self.param_group = create_group("Physics Control")
        par_lay = QtWidgets.QVBoxLayout(self.param_group)
        self.l_kappa = self._add_param_slider(par_lay, "L-Drift (κ):", 0, 100, 25)
        self.r_kappa = self._add_param_slider(par_lay, "R-Drift (κ):", 0, 100, 10)
        self.sidebar_layout.addWidget(self.param_group)

        self.sidebar_layout.addStretch()      
        self.sidebar_scroll.setWidget(self.sidebar)

        # 4. VIEWERS
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.view_left = MazeView(title="Mode A")
        self.view_right = MazeView(title="Mode B")
        self.viewer_splitter.addWidget(self.view_left)
        self.viewer_splitter.addWidget(self.view_right)

        self.main_layout.addWidget(self.sidebar_scroll)
        self.main_layout.addWidget(self.viewer_splitter)

        # 5. Connections
        self.maze_slider.valueChanged.connect(self.maze_spin.setValue)
        self.maze_spin.valueChanged.connect(self.maze_slider.setValue)
        self.maze_slider.valueChanged.connect(lambda v: self.mazeChanged.emit(v))
        
        for w in [self.l_kappa, self.r_kappa, self.model_selector, self.lens_selector, self.maze_slider]:
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(self.update_simulation)
            else: w.currentIndexChanged.connect(self.update_simulation)

        self.current_maze = None
        self.current_maze_jax = None
        self.rng_key = jax.random.PRNGKey(np.random.randint(1000))
        self.path_items = []

    def _add_param_slider(self, layout, label, min_v, max_v, start_v):
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt;")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(start_v)
        slider.setStyleSheet(self.slider_style)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider

    def set_dataset_name(self, name):
        self.ds_label.setText(f"Dataset: {name}")

    def set_maze(self, maze_np, idx=None):
        self.current_maze = maze_np
        self.current_maze_jax = jnp.array(maze_np)
        if idx is not None:
            self.maze_slider.blockSignals(True)
            self.maze_slider.setValue(idx)
            self.maze_spin.setValue(idx)
            self.maze_slider.blockSignals(False)
        self.update_simulation()

    def update_simulation(self):
        self.calc_timer.start(100)

    def _run_simulation_now(self):
        if self.current_maze_jax is None: return
        
        # Cleanup
        for view in [self.view_left, self.view_right]:
            view.clear_visuals(keep_image=False)
            for item in self.path_items: 
                try: view.removeItem(item)
                except: pass
        self.path_items = []

        # Params
        model_name = self.model_selector.currentText()
        policy = models.GeoPolicy() if "Geocentric" in model_name else models.EgoPolicy()
        l_params = {'kappa': self.l_kappa.value()/10.0, 'D_R': 0.15, 'goal_pos': jnp.array([15.0, 15.0])}
        r_params = {'kappa': self.r_kappa.value()/10.0, 'D_R': 0.15, 'goal_pos': jnp.array([15.0, 15.0])}

        self.rng_key, k1, k2 = jax.random.split(self.rng_key, 3)
        start_pos = jnp.array([0, 0])
        
        l_paths, l_occ = engine.simulate_swarm(k1, self.current_maze_jax, policy, l_params, start_pos)
        r_paths, r_occ = engine.simulate_swarm(k2, self.current_maze_jax, policy, r_params, start_pos)

        lens = self.lens_selector.currentText()
        self.view_left.set_maze(self.current_maze)
        self.view_right.set_maze(self.current_maze)

        if lens == "Probability Cloud":
            self.view_left.set_heatmap(self.current_maze, l_occ, cmap='viridis')
            self.view_right.set_heatmap(self.current_maze, r_occ, cmap='magma')
        elif lens == "Ghost Swarm":
            for i in range(15):
                self._draw_path(self.view_left, l_paths[i], '#28A745')
                self._draw_path(self.view_right, r_paths[i], '#00BFFF')
        elif lens == "Fluid Vector Field":
            field_l, _ = engine.compute_vector_field(self.current_maze_jax, policy, l_params)
            field_r, _ = engine.compute_vector_field(self.current_maze_jax, policy, r_params)
            self.view_left.draw_policy(self.current_maze, jnp.argmax(field_l, axis=-1), color='#28A745')
            self.view_right.draw_policy(self.current_maze, jnp.argmax(field_r, axis=-1), color='#00BFFF')

        self.view_left.setTitle(f"Left: κ={l_params['kappa']:.1f}", color='#28A745')
        self.view_right.setTitle(f"Right: κ={r_params['kappa']:.1f}", color='#00BFFF')

    def _draw_path(self, view, path_jax, color_hex):
        path = np.array(path_jax)
        curve = pg.PlotCurveItem(path[:,1], path[:,0], pen=pg.mkPen(pg.mkColor(color_hex + "44"), width=1.5))
        view.addItem(curve)
        self.path_items.append(curve)