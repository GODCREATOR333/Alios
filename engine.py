import os
import json
import numpy as np
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

        # --- Maze Navigation (Slider) ---
        self.maze_group = QtWidgets.QGroupBox("Maze Navigator")
        maze_layout = QtWidgets.QVBoxLayout(self.maze_group)
        self.maze_label = QtWidgets.QLabel("Maze Index: 0")
        self.maze_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.maze_slider.setRange(0, 999)
        maze_layout.addWidget(self.maze_label)
        maze_layout.addWidget(self.maze_slider)
        self.sidebar_layout.addWidget(self.maze_group)

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
        
        self.view_oracle = MazeView(title="Value Iteration (Oracle)")
        self.view_agent = MazeView(title="Agent Policy (Hypothesis)")
        
        self.viewer_layout.addWidget(self.view_oracle)
        self.viewer_layout.addWidget(self.view_agent)

        # Assemble Splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.viewer_widget)
        self.splitter.setSizes([300, 1100])
        self.main_layout.addWidget(self.splitter)

    def connect_signals(self):
        """Connects UI elements to logic methods."""
        self.refresh_runs_btn.clicked.connect(self.refresh_runs)
        self.run_selector.currentIndexChanged.connect(self.on_run_selected)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_selected)
        self.maze_slider.valueChanged.connect(self.on_maze_slider_moved)

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
        """Lists all .npy maze files in data_jax/."""
        self.dataset_selector.blockSignals(True)
        path = "data_jax"
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.npy')]
            self.dataset_selector.addItems(sorted(files))
        self.dataset_selector.blockSignals(False)

    def on_run_selected(self):
        """Triggered when a user picks a different training run."""
        run_id = self.run_selector.currentText()
        if not run_id: return
        
        details = db_manager.get_run_details(run_id)
        if details:
            # 1. Store state-representation and decoder function
            repr_key = details['state_repr']
            self.current_decoder = core_logic.DECODERS.get(repr_key, core_logic.decode_mdp)
            
            # 2. Update metadata display
            self.config_display.setText(json.dumps(details['config'], indent=4))
            
            # 3. Load the actual Q-table/Policy array
            if os.path.exists(details['path']):
                self.current_agent_q = np.load(details['path'])
                print(f"Loaded Agent Policy: {run_id} (Shape: {self.current_agent_q.shape})")
            
            self.update_display()

    def on_dataset_selected(self):
        ds_name = self.dataset_selector.currentText()
        if not ds_name: return
        
        # 1. Load the raw maze data
        ds_path = os.path.join("data_jax", ds_name)
        self.current_mazes = np.load(ds_path)
        
        # 2. Load the Oracle (.npz) baseline
        # Updated naming logic: .npy -> _VI_solved.npz
        vi_filename = ds_name.replace(".npy", "_VI_solved.npz")
        vi_path = os.path.join("data_jax", "value_iteration", vi_filename)
        
        if os.path.exists(vi_path):
            # Load the .npz dictionary
            with np.load(vi_path) as data:
                # We extract the policy and values into our "backpack"
                self.current_vi_policies = data['policy'] # Shape: (1000, 16, 16)
                self.current_vi_values = data['values']   # Shape: (1000, 16, 16)
            print(f"Loaded Oracle .npz: {vi_filename}")
        else:
            self.current_vi_policies = None
            self.current_vi_values = None
            print(f"Warning: No VI baseline found at {vi_path}")
            
        self.update_display()

    def on_maze_slider_moved(self):
        """Update label and redraw views when slider moves."""
        idx = self.maze_slider.value()
        self.maze_label.setText(f"Maze Index: {idx}")
        self.update_display()

    def update_display(self):
        idx = self.maze_slider.value()
        if self.current_mazes is None: return
        
        maze = self.current_mazes[idx]
        
        # 1. Update Oracle View
        self.view_oracle.set_maze(maze)
        if self.current_vi_policies is not None:
            oracle_2d = self.current_vi_policies[idx]
            # Use base_color='#28A745' (Nice Green) for the Oracle
            self.view_oracle.draw_policy(maze, oracle_2d, core_logic.decode_mdp, base_color='#28A745')

        # 2. Update Agent View
        self.view_agent.set_maze(maze)
        if self.current_agent_q is not None and self.current_decoder is not None:
            oracle_2d = self.current_vi_policies[idx] if self.current_vi_policies is not None else None
            # Blue for normal, Red for divergence (handled automatically)
            self.view_agent.draw_policy(maze, self.current_agent_q, self.current_decoder, oracle_policy=oracle_2d)