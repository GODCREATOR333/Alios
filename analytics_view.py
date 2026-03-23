import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

class AnalyticsDashboard(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('#121212')
        
        # --- Create the 2x2 Grid of Scientific Plots ---
        # 1. Phase Transition Plot
        self.plot_phase = self.addPlot(title="Phase Transition: Success vs Density")
        self.plot_phase.addLegend(offset=(30, 30))
        
        # 2. Optimality Scatter
        self.plot_scatter = self.addPlot(title="Optimality Gap Scatter")
        
        self.nextRow()
        
        # 3. Efficiency Distribution
        self.plot_dist = self.addPlot(title="Path Efficiency Distribution")
        
        # 4. Conflict Analysis
        self.plot_conflict = self.addPlot(title="Conflict Severity vs Success")

        # Standardize Styling
        for p in [self.plot_phase, self.plot_scatter, self.plot_dist, self.plot_conflict]:
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis('left').setPen('#888')
            p.getAxis('bottom').setPen('#888')

    def clear_all(self):
        self.plot_phase.clear()
        self.plot_scatter.clear()
        self.plot_dist.clear()
        self.plot_conflict.clear()

    def update_dashboard(self, results, run_ids, ds_names):
        """Generates scientific plots from benchmarking data."""
        self.clear_all()
        
        # --- 1. PLOT PHASE TRANSITION (Success vs Density) ---
        for i, run_id in enumerate(run_ids):
            # Collect data points for this agent
            agent_data = [results[(run_id, ds)] for ds in ds_names]
            agent_data.sort(key=lambda x: x['density']) # Ensure X-axis is sorted
            
            x = [d['density'] for d in agent_data]
            y = [d['success'] for d in agent_data]
            
            color = pg.intColor(i)
            self.plot_phase.plot(x, y, pen=pg.mkPen(color, width=2), 
                                 symbol='o', symbolBrush=color, name=run_id)
        
        # Add the theoretical Percolation Limit (0.592)
        self.plot_phase.addLine(x=0.592, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=2))
        self.plot_phase.setLabel('bottom', 'Obstacle Density (p)')
        self.plot_phase.setLabel('left', 'Success Rate (%)')

        # --- 2. PLOT OPTIMALITY SCATTER ---
        for i, run_id in enumerate(run_ids):
            color = pg.intColor(i, alpha=100) # Faded dots for scatter
            all_agent_steps = []
            all_oracle_steps = []
            
            for ds in ds_names:
                d = results[(run_id, ds)]
                if d['oracle_steps'] is not None:
                    mask = d['reached_mask']
                    all_agent_steps.extend(d['steps'][mask])
                    all_oracle_steps.extend(d['oracle_steps'][mask])
            
            if all_agent_steps:
                self.plot_scatter.plot(all_oracle_steps, all_agent_steps, 
                                       pen=None, symbol='t', symbolSize=5, symbolBrush=color)
        
        # Add 1:1 Reference line
        self.plot_scatter.plot([0, 100], [0, 100], pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
        self.plot_scatter.setLabel('bottom', 'Oracle Path Length')
        self.plot_scatter.setLabel('left', 'Agent Path Length')

        # --- 3. PLOT EFFICIENCY HISTOGRAM ---
        for i, run_id in enumerate(run_ids):
            color = pg.intColor(i)
            all_ratios = []
            for ds in ds_names:
                d = results[(run_id, ds)]
                if d['oracle_steps'] is not None:
                    mask = d['reached_mask']
                    # Path Length Ratio: Agent / Oracle
                    ratios = d['steps'][mask] / d['oracle_steps'][mask]
                    all_ratios.extend(ratios)
            
            if all_ratios:
                y, x = np.histogram(all_ratios, bins=np.linspace(1, 4, 30))
                self.plot_dist.plot(x, y, stepMode="center", fillLevel=0, 
                                    fillBrush=(*color.getRgb()[:3], 50), pen=color)
        self.plot_dist.setLabel('bottom', 'Efficiency Ratio (1.0 = Optimal)')