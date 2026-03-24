import os
import json
import numpy as np
import jax.numpy as jnp
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from analytics_view import AnalyticsDashboard
import jax

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

        # --5. Sorted Map ---
        self.right_sort_map = np.arange(1000) # Default 1:1 mapping
        self.sort_enabled = False

    def init_ui(self):
        """Builds the layout and widgets (Modular approach)."""
        self.main_container = QtWidgets.QVBoxLayout(self)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        self.main_container.setSpacing(0)

        # --- 1. WORKSPACE SWITCHER HEADER ---
        self.header = QtWidgets.QWidget()
        self.header.setMinimumHeight(55)
        self.header.setStyleSheet("background-color: #1a1a1a; border-bottom: 1px solid #333;")
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 5, 10, 0)
        
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
        # 1. Create the Scroll Area
        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True) # Lets the inner widget expand
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.sidebar_scroll.setMinimumWidth(320) # Slightly wider to fit the scrollbar
        
        # 2. Sleek Dark Mode Scrollbar CSS
        self.sidebar_scroll.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: #121212;
                width: 8px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)

        # 3. Create the actual sidebar widget that holds the UI
        self.sidebar = QtWidgets.QWidget()
        self.sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 15, 10) # Extra right margin for scrollbar
        self.sidebar_layout.setSpacing(12)

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
        self.sidebar_scroll.setWidget(self.sidebar)

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
        self.splitter.addWidget(self.sidebar_scroll)
        self.splitter.addWidget(self.viewer_widget)
        self.splitter.setSizes([320, 1080])
        self.inspector_layout.addWidget(self.splitter)

        self.workspaces.addWidget(self.inspector_page) # Index 0

        # --- WORKSPACE 2: ANALYTICS LAB ---
        self.analytics_page = QtWidgets.QWidget()
        analytics_main_layout = QtWidgets.QHBoxLayout(self.analytics_page)
        self.analytics_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # 1. Lab Sidebar (Left)
        self.lab_sidebar = QtWidgets.QWidget()
        self.lab_sidebar.setMinimumWidth(300)
        lab_sidebar_layout = QtWidgets.QVBoxLayout(self.lab_sidebar)

        # Multi-Agent Selection
        self.agent_list_group = QtWidgets.QGroupBox("Select Agents to Benchmark")
        agent_list_layout = QtWidgets.QVBoxLayout(self.agent_list_group)
        self.agent_list_widget = QtWidgets.QListWidget()
        self.agent_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        agent_list_layout.addWidget(self.agent_list_widget)
        lab_sidebar_layout.addWidget(self.agent_list_group)

        # Multi-Dataset Selection
        self.lab_dataset_group = QtWidgets.QGroupBox("Select Datasets to Test")
        lab_dataset_layout = QtWidgets.QVBoxLayout(self.lab_dataset_group)
        self.dataset_list_widget = QtWidgets.QListWidget()
        self.dataset_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lab_dataset_layout.addWidget(self.dataset_list_widget)
        lab_sidebar_layout.addWidget(self.lab_dataset_group)

        # --- NEW: Scientific Lenses Selection ---
        self.lab_lens_group = QtWidgets.QGroupBox("Scientific Lenses")
        lab_lens_layout = QtWidgets.QVBoxLayout(self.lab_lens_group)
        self.lens_list_widget = QtWidgets.QListWidget()
        
        # Populate the available lenses
        for lens_name in['Success Rate vs Density', 'Optimality Scatter', 'Efficiency Distribution','Value Localization (Anderson)']:
            item = QtWidgets.QListWidgetItem(lens_name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked) # Check them by default
            self.lens_list_widget.addItem(item)
            
        lab_lens_layout.addWidget(self.lens_list_widget)
        lab_sidebar_layout.addWidget(self.lab_lens_group)

        # Execution Button
        self.btn_run_benchmarks = QtWidgets.QPushButton("RUN AGENT COMPARISON")
        self.btn_run_benchmarks.setMinimumHeight(50)
        self.btn_run_benchmarks.setStyleSheet("background-color: #00BFFF; color: black; font-weight: bold; border-radius: 5px;")
        lab_sidebar_layout.addWidget(self.btn_run_benchmarks)
        lab_sidebar_layout.addStretch()

        # 2. THE FIX: Create the Dashboard object FIRST
        self.plot_container = AnalyticsDashboard() 

        # 3. Assemble: Add items to splitter in order
        self.analytics_splitter.addWidget(self.lab_sidebar)
        self.analytics_splitter.addWidget(self.plot_container) # Now this exists!
        self.analytics_splitter.setSizes([300, 1100])
        
        analytics_main_layout.addWidget(self.analytics_splitter)
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

        # Redraw when sort is toggled
        self.sort_checkbox.stateChanged.connect(self.update_display)
        # Connect the Jump Bridge
        self.plot_container.mazeSelected.connect(self.jump_to_maze)

        # Connect ALL 6 views to the neuro probe
        for view in[self.view_left_policy, self.view_left_values, self.view_left_entropy,
                     self.view_right_policy, self.view_right_values, self.view_right_entropy]:
            view.cellClicked.connect(self.update_neuro_probe)\
        
        # Connect Lab Button
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

    def on_workspace_changed(self, index):
        self.workspaces.setCurrentIndex(index)
    
    def on_lens_toggled(self, item):
        """Instantly updates the visible plots when a checkbox is toggled."""
        active_lenses =[self.lens_list_widget.item(i).text() 
                         for i in range(self.lens_list_widget.count()) 
                         if self.lens_list_widget.item(i).checkState() == QtCore.Qt.Checked]
        
        # Tell the dashboard to re-draw using cached data
        self.plot_container.update_active_lenses(active_lenses)

    def jump_to_maze(self, dataset_name, maze_idx):
        """The Bridge: Teleports user from Lab to Inspector outlier."""
        # 1. Turn OFF sort so index 42 matches File Index 42
        self.sort_checkbox.setChecked(False)
        
        # 2. Switch Dataset Dropdown
        found_idx = self.dataset_selector.findText(dataset_name)
        if found_idx >= 0:
            self.dataset_selector.setCurrentIndex(found_idx)
        
        # 3. Set Slider directly
        self.maze_slider.setValue(int(maze_idx))
        
        # 4. Force Switch to Workspace 0 (Inspector)
        self.btn_inspector.setChecked(True)
        self.on_workspace_changed(0)

    # ============================
    # DATA & LOGIC METHODS
    # ============================

    def refresh_runs(self):
        runs = db_manager.get_all_runs()
        
        # 1. Existing Dropdown Logic (Inspector)
        items = ["Oracle (Value Iteration)"] + runs
        for selector in [self.left_selector, self.right_selector]:
            selector.blockSignals(True)
            selector.clear()
            selector.addItems(items)
            selector.blockSignals(False)

        # 2. NEW: Lab Multi-Select List
        self.agent_list_widget.clear()
        for run_id in runs:
            item = QtWidgets.QListWidgetItem(run_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.agent_list_widget.addItem(item)

    def scan_datasets(self):
        """Lists all .npy maze files in the inspector dropdown and lab list."""
        path = "data_jax"
        if os.path.exists(path):
            files = sorted([f for f in os.listdir(path) if f.endswith('.npy') and 'train' not in f]) # Only test files
            
            # 1. Inspector Dropdown
            self.dataset_selector.blockSignals(True)
            self.dataset_selector.clear()
            self.dataset_selector.addItems(files)
            self.dataset_selector.blockSignals(False)
            
            # 2. Lab Multi-Select List
            self.dataset_list_widget.blockSignals(True)
            self.dataset_list_widget.clear()
            for f in files:
                item = QtWidgets.QListWidgetItem(f)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Unchecked)
                self.dataset_list_widget.addItem(item)
            self.dataset_list_widget.blockSignals(False)
        
        # Trigger first load for the inspector
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
        """Silently pre-computes rollouts and generates the virtual sort map."""
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

        # --- GENERATE THE RANKING MAP ---
        if self.right_rollout_cache is not None and self.current_vi_values is not None:
            agent_steps = self.right_rollout_cache['steps']
            oracle_steps = np.abs(self.current_vi_values[:, 0, 0])
            success_mask = self.right_rollout_cache['success']
            
            # Metric for sorting: Optimality Ratio. 
            # If failed (loop), we give it a huge number (999) to put it at the very top.
            gaps = np.where(success_mask, agent_steps / (oracle_steps + 1e-6), 999.0)
            
            # Sort indices: Argsort goes ascending, so we use [::-1] to put worst first
            self.right_sort_map = np.argsort(gaps)[::-1]
        else:
            # Fallback to 1:1 mapping if no agent loaded
            self.right_sort_map = np.arange(1000)

    def run_batch_test(self):
        if self.right_q is None or self.current_mazes_jax is None:
            return
            
        self.run_test_btn.setText("Computing Rollouts...")
        QtWidgets.QApplication.processEvents()
        
        details = db_manager.get_run_details(self.right_selector.currentText())
        r_key = details['state_repr']
        map_func = core_logic.STATE_MAP_FUNCS_RAW.get(r_key, core_logic.STATE_MAP_FUNCS_RAW['mdp'])

        res = core_logic.evaluate_dataset(self.right_q, self.current_mazes_jax, map_func)
        
        steps_arr = np.array(res[2])
        colls_arr = np.array(res[3])
        goal_arr = np.array(res[4])
        
        total = len(goal_arr)
        success_count = np.sum(goal_arr)
        success_rate = (success_count / total) * 100
        timeout_rate = (np.sum(steps_arr >= 500) / total) * 100
        
        # --- THE FIX: GLOBAL STATS ---
        # We calculate collisions across ALL 1,000 mazes
        global_avg_colls = np.mean(colls_arr) 
        
        # Efficiency stats (Successful runs only)
        avg_steps_success = 0
        opt_gap = 1.0

        # Calculate how many arrows in the entire dataset point to walls
        # We can vmap the new safety function over the batch
        safety_scores = jax.vmap(core_logic.calculate_policy_safety, in_axes=(None, 0, None))(
            self.right_q, self.current_mazes_jax, self.right_decoder_raw)
        avg_field_unsafe = np.mean(safety_scores)

        if success_count > 0:
            success_mask = (goal_arr == True)
            avg_steps_success = np.mean(steps_arr[success_mask])
            
            if self.current_vi_policies is not None:
                oracle_steps_all = np.abs(self.current_vi_values[:, 0, 0])
                avg_oracle = np.mean(oracle_steps_all[success_mask])
                opt_gap = avg_steps_success / (avg_oracle + 1e-6)

        # 4. Updated Display Result
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
        """Orchestrates the (Agents x Datasets) statistical experiment."""
        # 1. Gather Selected Agents from the list checkboxes
        selected_runs = [self.agent_list_widget.item(i).text() 
                         for i in range(self.agent_list_widget.count()) 
                         if self.agent_list_widget.item(i).checkState() == QtCore.Qt.Checked]
        
        # 2. Gather Selected Datasets from the list checkboxes
        selected_datasets = [self.dataset_list_widget.item(i).text() 
                             for i in range(self.dataset_list_widget.count()) 
                             if self.dataset_list_widget.item(i).checkState() == QtCore.Qt.Checked]

        if not selected_runs or not selected_datasets:
            self.stats_display.setText("<b style='color:red;'>Error:</b> Select at least one agent and one dataset.")
            return

        # UI Feedback to show work is happening
        self.btn_run_benchmarks.setText("Processing Experiment...")
        self.btn_run_benchmarks.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        master_results = {}

        # 3. The Grand Loop: Datasets x Agents
        for ds_name in selected_datasets:
            # Load Dataset and Oracle for this specific file
            ds_path = os.path.join("data_jax", ds_name)
            mazes_jax = jnp.array(np.load(ds_path))
            
            # Fetch Oracle for steps comparison
            vi_filename = ds_name.replace(".npy", "_VI_solved.npz")
            vi_path = os.path.join("data_jax", "value_iteration", vi_filename)
            oracle_v_start = None
            if os.path.exists(vi_path):
                with np.load(vi_path) as vi_data:
                    # Value at (0,0) is negative path length
                    oracle_v_start = np.abs(vi_data['values'][:, 0, 0])

            for run_id in selected_runs:
                details = db_manager.get_run_details(run_id)
                if not details: continue
                
                # Load Agent Data
                q_table = jnp.array(np.load(details['path']))
                r_key = details['state_repr']
                
                # Use RAW mapper for JAX Batch Evaluation
                map_func = core_logic.STATE_MAP_FUNCS_RAW.get(r_key, core_logic.STATE_MAP_FUNCS_RAW['mdp'])

                # Run High-Speed JAX Evaluation
                res = core_logic.evaluate_dataset(q_table, mazes_jax, map_func)
                reached = np.array(res[4])
                steps = np.array(res[2])
                
                # Calculate Success Rate
                success_rate = (np.sum(reached) / len(reached)) * 100

                # --- NEW PHYSICS PROBE ---
                pr_scores = core_logic.calculate_localization_batch(q_table, mazes_jax, map_func)
                avg_pr = np.mean(pr_scores)
                # -------------------------
                
                # Extract Obstacle Density from filename (e.g., P0100 -> 0.1)
                try:
                    density = float(ds_name.split('_P')[1].split('_')[0]) / 1000.0
                except:
                    density = 0.0

                # Store all raw data for the Dashboard to plot
                master_results[(run_id, ds_name)] = {
                    'success': success_rate,
                    'pr': avg_pr,
                    'density': density,
                    'reached_mask': reached,
                    'steps': steps,
                    'oracle_steps': oracle_v_start
                }

        # --- THE MODULAR HANDOVER ---
        # 1. Save data to Cache
        self.plot_container.set_data_cache(master_results, selected_runs, selected_datasets)
        
        # 2. Trigger the UI to draw whatever is currently checked
        self.on_lens_toggled(None)
        
        self.btn_run_benchmarks.setText("RUN AGENT COMPARISON")
        self.btn_run_benchmarks.setEnabled(True)

    def get_real_maze_idx(self):
        """Translates the current slider rank into the actual dataset index."""
        slider_val = self.maze_slider.value()
        # If sorting is enabled, look up the real index in the sort map
        if hasattr(self, 'sort_checkbox') and self.sort_checkbox.isChecked() and hasattr(self, 'right_sort_map'):
            return int(self.right_sort_map[slider_val])
        # Otherwise, the slider value is the index
        return slider_val

    def update_neuro_probe(self, r, c, state_id_argument, count_argument):
        # --- 1. THE RESET FIX (Toggle Off) ---
        if r == -1:
            self.neuro_label.setText("Click a cell to probe its internal state.")
            self.inspect_group.setTitle("Neuro-Probe")
            for view in [self.view_left_policy, self.view_left_values, self.view_left_entropy,
                         self.view_right_policy, self.view_right_values, self.view_right_entropy]:
                view.clear_highlights()
                view.clear_trajectory()
            return

        # 2. Get correct indices for the current maze (Supports Sorting)
        idx = self.get_real_maze_idx()
        maze_jax = self.current_mazes_jax[idx]
        maze_numpy = self.current_mazes[idx]

        # --- 3. DETERMINE THE ACTIVE "EYES" AND "BRAIN" ---
        # Priority: Right Agent > Left Agent > Default MDP
        if self.right_type == 'agent' and self.right_q is not None:
            active_jit = self.right_decoder_jit
            active_scalar = self.right_decoder
            active_q = self.right_q
        elif self.left_type == 'agent' and self.left_q is not None:
            active_jit = self.left_decoder_jit
            active_scalar = self.left_decoder
            active_q = self.left_q
        else:
            active_jit = core_logic.STATE_MAP_FUNCS_JIT['mdp']
            active_scalar = core_logic.decode_mdp
            active_q = None

        if active_jit is not None:
            # --- 4. THE FIRST-CLICK FIX: RECALCULATE STATE ID ---
            # We calculate the ID here in the engine using the active agent's logic.
            # This makes the click logic independent of which panel was clicked.
            state_id = int(active_scalar(maze_numpy, r, c))
            
            # Calculate the shared map to know where the aliasing is
            shared_map_jax = active_jit(maze_jax)
            shared_map_np = np.array(shared_map_jax)
            
            # Recalculate true aliased count
            aliased_count = int(np.sum(shared_map_np == state_id))
            
            # 5. SYNC COLORS: Left is Blue, Right is Gold
            blue_rgba, gold_rgba = [0, 191, 255, 120], [255, 215, 0, 120]

            # Update ALL 6 tabs with the fresh map and calculated State ID
            for v in [self.view_left_policy, self.view_left_values, self.view_left_entropy]:
                v.set_state_map(shared_map_np)
                v.highlight_aliased_states(state_id, blue_rgba)
            
            for v in [self.view_right_policy, self.view_right_values, self.view_right_entropy]:
                v.set_state_map(shared_map_np)
                v.highlight_aliased_states(state_id, gold_rgba)

            # 6. MATH & TRAJECTORY (Only if we have an agent brain)
            if active_q is not None:
                q_vals = active_q[state_id]
                entropy = core_logic.calculate_entropy(q_vals)
                
                conflict = 0.0
                if self.current_vi_policies is not None:
                    oracle_p = self.current_vi_policies[idx]
                    conflict = core_logic.calculate_conflict(state_id, shared_map_np, oracle_p)
                
                # Update Sidebar Text
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

                # Update Trajectory Line
                path = core_logic.compute_rollout(maze_numpy, (r, c), active_q, active_scalar)
                self.view_left_policy.draw_trajectory(path)
                self.view_right_policy.draw_trajectory(path)
            else:
                self.neuro_label.setText("Probing Oracle (MDP mode).<br>No Q-values available.")

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
        """Symmetrical update logic for all panels using a configuration loop."""
        idx = self.maze_slider.value()
        if self.current_mazes is None:
            return
        
        # --- THE TRANSLATION BRIDGE ---
        slider_val = self.maze_slider.value()
        
        # If sort is checked, 'idx' becomes the mapped rank. 
        # Otherwise, 'idx' is just the slider number.
        if self.sort_checkbox.isChecked() and hasattr(self, 'right_sort_map'):
            idx = int(self.right_sort_map[slider_val])
            self.maze_label.setText(f"Rank: {slider_val} (Maze #{idx})")
        else:
            idx = slider_val
            self.maze_label.setText(f"Maze Index: {idx}")
        # -------------------------------
        
        maze = self.current_mazes[idx]
        maze_jax = self.current_mazes_jax[idx]
        
        # Shortest path steps from Oracle (Value at 0,0 is -steps)
        oracle_steps = int(abs(self.current_vi_values[idx, 0, 0])) if self.current_vi_values is not None else 0

        # 1. Fetch data for both sides using the helper
        L_act, L_val, L_ent, L_map = self._get_panel_data(self.left_type, self.left_q, self.left_decoder_jit, idx, maze_jax)
        R_act, R_val, R_ent, R_map = self._get_panel_data(self.right_type, self.right_q, self.right_decoder_jit, idx, maze_jax)

        # 2. Define the Panel Configurations
        # Format: (SideName, Type, (Actions, Values, Entropy, StateMap), [Tabs], BaseColor, Cache)
        panel_configs = [
            ('Left', self.left_type, (L_act, L_val, L_ent, L_map), 
             [self.view_left_policy, self.view_left_values, self.view_left_entropy], '#28A745', self.left_rollout_cache),
            
            ('Right', self.right_type, (R_act, R_val, R_ent, R_map), 
             [self.view_right_policy, self.view_right_values, self.view_right_entropy], '#00BFFF', self.right_rollout_cache)
        ]

        # 3. Loop through and update both panels
        for name, p_type, data, views, color, cache in panel_configs:
            act, val, ent, smap = data
            
            # Reset views and clear previous highlights
            for v in views:
                v.set_maze(maze)
                # IMPORTANT: We give every view the state map for the probe
                if smap is not None:
                    v.set_state_map(smap)

            # Draw the layers if data exists
            if act is not None:
                # DIVERGENCE: Right panel compares itself to Left Panel (L_act)
                comparison_baseline = L_act if name == 'Right' else None
                
                views[0].draw_policy_vectorized(maze, act, oracle_actions=comparison_baseline, base_color=color)
                views[1].set_heatmap(maze, val)
                views[2].set_heatmap(maze, ent)
            else:
                # Tell the viewer to hide all arrows
                views[0].draw_policy_vectorized(maze, np.zeros((16,16)), base_color=color)
                # Ensure heatmap images are cleared
                views[1].img.clear()
                views[2].img.clear()

            # 4. Update the titles (Scoreboard)
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
            # Save the right-side title so the Neuro-Probe can restore it later
            if name == 'Right':
                self.current_maze_score = title