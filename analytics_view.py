import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PyQt5.QtCore import pyqtSignal

class AnalyticsDashboard(QtWidgets.QScrollArea):
    mazeSelected = pyqtSignal(str, int)
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet("background-color: #121212;")

        # Container for the vertically stacked plots
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.container)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)

        # --- THE LENS REGISTRY ---
        # Every new analysis tool goes into this dictionary
        self.lenses = {
            'Success Rate vs Density': pg.PlotWidget(title="Phase Transition: Success vs Density"),
            'Optimality Scatter': pg.PlotWidget(title="Optimality Gap Scatter"),
            'Efficiency Distribution': pg.PlotWidget(title="Path Efficiency Distribution"),
            'Value Localization (Anderson)': pg.PlotWidget(title="Value Localization (Anderson Index)")
        }

        # Style and add them all, but HIDE them by default
        for name, plot in self.lenses.items():
            plot.setMinimumHeight(350) # Big enough to see clearly
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.getAxis('left').setPen('#888')
            plot.getAxis('bottom').setPen('#888')
            self.layout.addWidget(plot)
            plot.hide()

        self.layout.addStretch()
        self.setWidget(self.container)

        # --- THE DATA CACHE ---
        self.cached_results = None
        self.cached_runs = []
        self.cached_datasets =[]



    def set_data_cache(self, results, run_ids, ds_names):
        """Stores the massive JAX rollout data in memory."""
        self.cached_results = results
        self.cached_runs = run_ids
        self.cached_datasets = ds_names

    def update_active_lenses(self, active_lens_names):
        """Instantly hides/shows/redraws plots based on UI checkboxes."""
        if not self.cached_results: 
            return

        # Hide all first
        for plot in self.lenses.values():
            plot.hide()
            plot.clear() # Clear old data

        # Render and show only the checked ones
        if 'Success Rate vs Density' in active_lens_names:
            self._draw_success_lens(self.lenses['Success Rate vs Density'])
            self.lenses['Success Rate vs Density'].show()
            
        if 'Optimality Scatter' in active_lens_names:
            self._draw_optimality_lens(self.lenses['Optimality Scatter'])
            self.lenses['Optimality Scatter'].show()

        if 'Efficiency Distribution' in active_lens_names:
            self._draw_efficiency_lens(self.lenses['Efficiency Distribution'])
            self.lenses['Efficiency Distribution'].show()

        if 'Value Localization (Anderson)' in active_lens_names:
            self._draw_localization_lens(self.lenses['Value Localization (Anderson)'])
            self.lenses['Value Localization (Anderson)'].show()

    # --- INDIVIDUAL LENS RENDERING LOGIC ---

    def _draw_success_lens(self, plot):
        plot.addLegend(offset=(30, 30))
        for i, run_id in enumerate(self.cached_runs):
            agent_data = [self.cached_results[(run_id, ds)] for ds in self.cached_datasets]
            agent_data.sort(key=lambda x: x['density'])
            x = [d['density'] for d in agent_data]
            y = [d['success'] for d in agent_data]
            color = pg.intColor(i)
            plot.plot(x, y, pen=pg.mkPen(color, width=2), symbol='o', symbolBrush=color, name=run_id)
        
        plot.addLine(x=0.592, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=2)) # Percolation Threshold
        plot.setLabel('bottom', 'Obstacle Density (p)')
        plot.setLabel('left', 'Success Rate (%)')

    def _draw_optimality_lens(self, plot):
        """Redraws the scatter plot and makes points interactive."""
        plot.addLegend(offset=(30, 30))
        
        for i, run_id in enumerate(self.cached_runs):
            color = pg.intColor(i, alpha=150)
            for ds in self.cached_datasets:
                d = self.cached_results[(run_id, ds)]
                if d['oracle_steps'] is not None:
                    reached = d['reached_mask']
                    
                    # Create a ScatterPlotItem for high performance and clicking
                    scatter = pg.ScatterPlotItem(
                        size=8, pen=pg.mkPen(None), brush=color, 
                        name=f"{run_id} ({ds})", hoverable=True
                    )
                    
                    # Store indices and dataset names in the 'data' field of each point
                    indices = np.where(reached)[0]
                    points = []
                    for idx_in_batch in range(len(indices)):
                        actual_idx = indices[idx_in_batch]
                        points.append({
                            'pos': (d['oracle_steps'][actual_idx], d['steps'][actual_idx]),
                            'data': (ds, int(actual_idx)) # (Dataset name, Maze index)
                        })
                    
                    scatter.addPoints(points)
                    plot.addItem(scatter)
                    
                    # Connect the click signal for this specific agent's dots
                    scatter.sigClicked.connect(self._on_scatter_clicked)
        
        plot.plot([0, 100], [0, 100], pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
        plot.setLabel('bottom', 'Oracle Path Length')
        plot.setLabel('left', 'Agent Path Length')

    def _on_scatter_clicked(self, scatter_item, points):
        """Handles clicking a dot in the scatter plot."""
        # Use len() to check for empty lists to avoid NumPy ambiguity errors
        if len(points) == 0: 
            return
            
        # Get the metadata from the FIRST point clicked
        try:
            dataset_name, maze_idx = points[0].data()
            # Emit signal to the engine
            self.mazeSelected.emit(dataset_name, int(maze_idx))
        except Exception as e:
            print(f"Jump Error: {e}")

    def _draw_efficiency_lens(self, plot):
        for i, run_id in enumerate(self.cached_runs):
            color = pg.intColor(i)
            all_ratios =[]
            for ds in self.cached_datasets:
                d = self.cached_results[(run_id, ds)]
                if d['oracle_steps'] is not None:
                    mask = d['reached_mask']
                    all_ratios.extend(d['steps'][mask] / d['oracle_steps'][mask])
            if all_ratios:
                y, x = np.histogram(all_ratios, bins=np.linspace(1, 4, 30))
                plot.plot(x, y, stepMode="center", fillLevel=0, fillBrush=(*color.getRgb()[:3], 50), pen=color)
        plot.setLabel('bottom', 'Efficiency Ratio (Steps / Oracle)')

    def _draw_localization_lens(self, plot):
        plot.addLegend(offset=(30, 30))
        for i, run_id in enumerate(self.cached_runs):
            agent_data = [self.cached_results[(run_id, ds)] for ds in self.cached_datasets]
            agent_data.sort(key=lambda x: x['density'])
            
            x = [d['density'] for d in agent_data]
            y = [d['pr'] for d in agent_data] # The PR index
            
            color = pg.intColor(i)
            plot.plot(x, y, pen=pg.mkPen(color, width=2), symbol='s', name=run_id)
            
        plot.setLabel('bottom', 'Obstacle Density (p)')
        plot.setLabel('left', 'Localization Index (PR)')