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
        
        # --- MODERN FLAT CSS ---
        self.setStyleSheet("""
            QWidget { background-color: #0d0d0d; color: #d4d4d4; }
            QGroupBox { 
                background-color: #171717; 
                border-radius: 8px; 
                margin-top: 10px; 
                padding: 15px 10px 10px 10px; 
            }
            QComboBox { 
                background-color: #222; border: 1px solid #333; 
                border-radius: 4px; padding: 4px; 
            }
            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { background: #00BFFF; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
        """)

        self.calc_timer = QtCore.QTimer()
        self.calc_timer.setSingleShot(True)
        self.calc_timer.timeout.connect(self._run_simulation_now)

        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- SLEEK SIDEBAR ---
        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFixedWidth(280)
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar_scroll.setStyleSheet("background-color: #0d0d0d; border-right: 1px solid #222;")
        
        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(15, 20, 15, 20)
        self.sidebar_layout.setSpacing(15)

        # 1. Header & Navigator
        header = QtWidgets.QLabel("DYNAMICS LAB")
        header.setStyleSheet("color: #fff; font-size: 14pt; font-weight: bold; letter-spacing: 2px;")
        self.sidebar_layout.addWidget(header)

        self.ds_label = QtWidgets.QLabel("DS: None")
        self.ds_label.setStyleSheet("color: #666; font-size: 9pt;")
        self.sidebar_layout.addWidget(self.ds_label)

        nav_row = QtWidgets.QHBoxLayout()
        self.maze_spin = QtWidgets.QSpinBox()
        self.maze_spin.setRange(0, 999)
        self.maze_spin.setFixedWidth(60)
        self.maze_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.maze_spin.setStyleSheet("background: #222; color: #00BFFF; border-radius: 4px; padding: 4px;")
        
        self.maze_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maze_slider.setRange(0, 999)
        
        nav_row.addWidget(QtWidgets.QLabel("ID:"))
        nav_row.addWidget(self.maze_spin)
        nav_row.addWidget(self.maze_slider)
        self.sidebar_layout.addLayout(nav_row)

        # 2. Core Settings Card
        settings_card = QtWidgets.QGroupBox()
        set_lay = QtWidgets.QVBoxLayout(settings_card)
        
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems(["Geocentric (Goal Drift)", "Egocentric (Active Matter)"])
        self.lens_selector = QtWidgets.QComboBox()
        self.lens_selector.addItems(["Probability Cloud", "Ghost Swarm", "Fluid Vector Field"])
        
        set_lay.addWidget(QtWidgets.QLabel("Policy Architecture:"))
        set_lay.addWidget(self.model_selector)
        set_lay.addSpacing(10)
        set_lay.addWidget(QtWidgets.QLabel("Visualization Lens:"))
        set_lay.addWidget(self.lens_selector)
        self.sidebar_layout.addWidget(settings_card)

        # 3. Left Params Card
        l_card = QtWidgets.QGroupBox()
        l_lay = QtWidgets.QVBoxLayout(l_card)
        l_title = QtWidgets.QLabel("■ MODE A PARAMETERS")
        l_title.setStyleSheet("color: #28A745; font-weight: bold;")
        l_lay.addWidget(l_title)
        self.l_kappa = self._add_slider(l_lay, "Goal Drift (κ):", 0, 100, 25, "#28A745")
        self.l_dr = self._add_slider(l_lay, "Rotational Inertia (Dr):", 0, 100, 15, "#28A745")
        self.sidebar_layout.addWidget(l_card)

        # 4. Right Params Card
        r_card = QtWidgets.QGroupBox()
        r_lay = QtWidgets.QVBoxLayout(r_card)
        r_title = QtWidgets.QLabel("■ MODE B PARAMETERS")
        r_title.setStyleSheet("color: #00BFFF; font-weight: bold;")
        r_lay.addWidget(r_title)
        self.r_kappa = self._add_slider(r_lay, "Goal Drift (κ):", 0, 100, 10, "#00BFFF")
        self.r_dr = self._add_slider(r_lay, "Rotational Inertia (Dr):", 0, 100, 50, "#00BFFF")
        self.sidebar_layout.addWidget(r_card)

        self.sidebar_layout.addStretch()      
        self.sidebar_scroll.setWidget(self.sidebar)

        # --- VIEWERS ---
        self.viewer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.view_left = MazeView(title="Mode A")
        self.view_right = MazeView(title="Mode B")
        self.viewer_splitter.addWidget(self.view_left)
        self.viewer_splitter.addWidget(self.view_right)

        self.main_layout.addWidget(self.sidebar_scroll)
        self.main_layout.addWidget(self.viewer_splitter)

        # --- CONNECTIONS ---
        self.maze_slider.valueChanged.connect(self.maze_spin.setValue)
        self.maze_spin.valueChanged.connect(self.maze_slider.setValue)
        self.maze_slider.valueChanged.connect(lambda v: self.mazeChanged.emit(v))
        
        for w in [self.l_kappa, self.l_dr, self.r_kappa, self.r_dr, self.model_selector, self.lens_selector]:
            if hasattr(w, 'valueChanged'): w.valueChanged.connect(self._trigger_update)
            else: w.currentIndexChanged.connect(self._trigger_update)

        self.current_maze = None
        self.current_maze_jax = None
        self.rng_key = jax.random.PRNGKey(np.random.randint(1000))
        self.path_items = []

    def _add_slider(self, layout, label, min_v, max_v, start_v, color):
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 8pt;")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(start_v)
        slider.setStyleSheet(f"QSlider::handle:horizontal {{ background: {color}; }} QSlider::sub-page:horizontal {{ background: {color}; }}")
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider

    def set_dataset_name(self, name):
        self.ds_label.setText(f"DS: {name}")

    def set_maze(self, maze_np, idx=None):
        self.current_maze = maze_np
        self.current_maze_jax = jnp.array(maze_np)
        if idx is not None:
            self.maze_slider.blockSignals(True)
            self.maze_slider.setValue(idx)
            self.maze_spin.setValue(idx)
            self.maze_slider.blockSignals(False)
        self._trigger_update()

    def _trigger_update(self):
        # Debounce: wait 150ms after user STOPS dragging slider before computing
        self.calc_timer.start(150)

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
        
        l_params = {'kappa': self.l_kappa.value()/10.0, 'D_R': self.l_dr.value()/100.0, 'goal_pos': jnp.array([15.0, 15.0])}
        r_params = {'kappa': self.r_kappa.value()/10.0, 'D_R': self.r_dr.value()/100.0, 'goal_pos': jnp.array([15.0, 15.0])}

        # Simulate
        self.rng_key, k1, k2 = jax.random.split(self.rng_key, 3)
        start_pos = jnp.array([0, 0])
        
        l_paths, l_occ = engine.simulate_swarm(k1, self.current_maze_jax, policy, l_params, start_pos)
        r_paths, r_occ = engine.simulate_swarm(k2, self.current_maze_jax, policy, r_params, start_pos)

        lens = self.lens_selector.currentText()
        self.view_left.set_maze(self.current_maze)
        self.view_right.set_maze(self.current_maze)

        # --- THE SMOOTH HEATMAP TRICK (Log Scaling) ---
        if lens == "Probability Cloud":
            # Applying log1p smooths out the spikey step-counts into a glowing cloud
            l_smooth = np.log1p(np.array(l_occ))
            r_smooth = np.log1p(np.array(r_occ))
            self.view_left.set_heatmap(self.current_maze, l_smooth, cmap='viridis')
            self.view_right.set_heatmap(self.current_maze, r_smooth, cmap='magma')
            
        elif lens == "Ghost Swarm":
            for i in range(15):
                self._draw_path(self.view_left, l_paths[i], '#28A745')
                self._draw_path(self.view_right, r_paths[i], '#00BFFF')
                
        elif lens == "Fluid Vector Field":
            field_l, _ = engine.compute_vector_field(self.current_maze_jax, policy, l_params)
            field_r, _ = engine.compute_vector_field(self.current_maze_jax, policy, r_params)
            self.view_left.draw_policy(self.current_maze, jnp.argmax(field_l, axis=-1), color='#28A745')
            self.view_right.draw_policy(self.current_maze, jnp.argmax(field_r, axis=-1), color='#00BFFF')

        self.view_left.setTitle(f"Mode A │ κ={l_params['kappa']:.1f} │ Dr={l_params['D_R']:.2f}", color='#28A745')
        self.view_right.setTitle(f"Mode B │ κ={r_params['kappa']:.1f} │ Dr={r_params['D_R']:.2f}", color='#00BFFF')

    def _draw_path(self, view, path_jax, color_hex):
        path = np.array(path_jax)
        curve = pg.PlotCurveItem(path[:,1], path[:,0], pen=pg.mkPen(pg.mkColor(color_hex + "66"), width=1.5))
        view.addItem(curve)
        self.path_items.append(curve)