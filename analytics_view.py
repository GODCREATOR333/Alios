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
        
        # --- Ultra-Clean White/Glass Scrollbar ---
        self.setStyleSheet("""
            QScrollArea { background-color: #0b0b0b; border: none; }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 0.05);
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #e0e0e0;
                min-height: 40px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #ffffff; }
        """)

        self.container = QtWidgets.QWidget()
        self.container.setStyleSheet("background-color: #0b0b0b;")
        self.layout = QtWidgets.QVBoxLayout(self.container)
        self.layout.setSpacing(60)
        self.layout.setContentsMargins(60, 40, 60, 60)

        # --- Scientific Gradient Palettes ---
        # Each agent gets a "Family" of 3 shades
        self.palettes = [
            ['#1f77b4', '#4aa3df', '#a2d2ff'], # Blues
            ['#ff7f0e', '#ffbb78', '#ffe9a6'], # Oranges
            ['#2ca02c', '#98df8a', '#c1f0c1'], # Greens
            ['#d62728', '#ff9896', '#ffccd5'], # Reds
            ['#9467bd', '#c5b0d5', '#e8def1'], # Purples
            ['#00ced1', '#7fffd4', '#e0ffff'], # Cyans
        ]

        self.lenses = {}
        self._init_plots()
        
        self.layout.addStretch()
        self.setWidget(self.container)
        self.cached_results = None

    def _init_plots(self):
        configs = [
            ('Success Rate vs Density', "Phase Transition: Navigation Integrity", "Obstacle Density (ρ)", "Success Rate (%)"),
            ('Optimality Scatter', "Path Length Analysis", "Optimal Steps (Oracle)", "Agent Steps"),
            ('Anderson Localization (IPR)', "Agent State Localization (Anderson Index)", "Obstacle Density (ρ)", "IPR (Lower = More Extended)"),
            ('Mean Squared Displacement', "Spatial Diffusion (MSD Profile)", "Obstacle Density (ρ)", "MSD <Δr²>")
        ]
        
        for key, title, x, y in configs:
            p = pg.PlotWidget()
            p.setMinimumHeight(500)
            
            # Remove default border and style axes
            p.showGrid(x=True, y=True, alpha=0.1)
            p.getAxis('bottom').setLabel(x, color='#777', size='12pt')
            p.getAxis('left').setLabel(y, color='#777', size='12pt')
            
            # Title Styling: Large, Clean, Muted White
            p.setTitle(title, color='#eee', size='14pt')
            
            p.setAntialiasing(True)
            self.lenses[key] = p
            self.layout.addWidget(p)
            p.hide()

    def set_data_cache(self, results, run_ids, ds_names):
        self.cached_results = results
        self.cached_runs = run_ids
        self.cached_datasets = ds_names

    def update_active_lenses(self, active_lens_names):
        if not self.cached_results: return
        for name, plot in self.lenses.items():
            if name in active_lens_names:
                plot.show()
                plot.clear()
                # Legend with a soft glass background
                plot.addLegend(offset=(20, 20), labelTextColor="#ccc", brush=(20,20,20,180), pen=(100,100,100))
                self._render_lens(name, plot)
            else:
                plot.hide()

    def _render_lens(self, name, plot):
        if name == 'Success Rate vs Density': self._draw_success(plot)
        elif name == 'Optimality Scatter': self._draw_optimality(plot)
        elif name == 'Anderson Localization (IPR)': self._draw_ipr(plot)
        elif name == 'Mean Squared Displacement': self._draw_msd(plot)

    def _draw_success(self, plot):
        for i, run_id in enumerate(self.cached_runs):
            palette = self.palettes[i % len(self.palettes)]
            agent_data = [self.cached_results[(run_id, ds)] for ds in self.cached_datasets]
            categories = sorted(list(set([d['category'] for d in agent_data])))
            
            for j, cat in enumerate(categories):
                cat_color = palette[j % len(palette)]
                cat_data = sorted([d for d in agent_data if d['category'] == cat], key=lambda x: x['density'])
                
                x = [d['density'] for d in cat_data]
                y = [d['success'] for d in cat_data]
                
                # --- THE GLOW EFFECT ---
                # Draw a faint thicker line behind the main line
                glow_pen = pg.mkPen(cat_color, width=6)
                glow_pen.setCapStyle(QtCore.Qt.RoundCap)
                glow_color = pg.mkColor(cat_color)
                glow_color.setAlpha(40)
                plot.plot(x, y, pen=pg.mkPen(glow_color, width=8))

                # --- THE MAIN DATA LINE ---
                line_name = f"{run_id} | {cat.capitalize()}"
                plot.plot(x, y, pen=pg.mkPen(cat_color, width=2.5), 
                          symbol='o', symbolSize=8, symbolBrush=cat_color, 
                          symbolPen=pg.mkPen(None), name=line_name)

        # Percolation Reference Line
        ref = pg.InfiniteLine(pos=0.592, angle=90, pen=pg.mkPen('#cc4444', width=1.5, style=QtCore.Qt.DashLine))
        plot.addItem(ref)

    def _draw_optimality(self, plot):
        for i, run_id in enumerate(self.cached_runs):
            base_color = self.palettes[i % len(self.palettes)][0]
            color = pg.mkColor(base_color)
            color.setAlpha(100)
            
            scatter = pg.ScatterPlotItem(size=9, brush=color, pen=pg.mkPen(None), hoverable=True)
            points = []
            
            for ds in self.cached_datasets:
                d = self.cached_results[(run_id, ds)]
                if d['oracle_steps'] is not None:
                    mask = d['reached_mask']
                    for idx in np.where(mask)[0]:
                        points.append({
                            'pos': (d['oracle_steps'][idx], d['steps'][idx]),
                            'data': (ds, int(idx))
                        })
            scatter.addPoints(points)
            scatter.sigClicked.connect(self._on_scatter_clicked)
            plot.addItem(scatter)
        
        # Diagonal reference
        plot.plot([0, 100], [0, 100], pen=pg.mkPen('#333', width=1, style=QtCore.Qt.DashLine))

    def _draw_ipr(self, plot):
        for i, run_id in enumerate(self.cached_runs):
            color = self.palettes[i % len(self.palettes)][0]
            data = sorted([self.cached_results[(run_id, ds)] for ds in self.cached_datasets], key=lambda x: x['density'])
            
            unique_x = sorted(list(set([d['density'] for d in data])))
            # Error-bar logic: plot the mean and the spread
            means = [np.mean([d['ipr'] for d in data if d['density'] == ux]) for ux in unique_x]
            
            # Glow Line
            p_glow = pg.mkColor(color)
            p_glow.setAlpha(30)
            plot.plot(unique_x, means, pen=pg.mkPen(p_glow, width=10))
            
            # Sharp Core Line
            plot.plot(unique_x, means, pen=pg.mkPen(color, width=3, style=QtCore.Qt.DashLine), 
                      symbol='s', symbolSize=10, symbolBrush=color, name=run_id)

    def _draw_msd(self, plot):
        for i, run_id in enumerate(self.cached_runs):
            color = self.palettes[i % len(self.palettes)][0]
            data = sorted([self.cached_results[(run_id, ds)] for ds in self.cached_datasets], key=lambda x: x['density'])
            
            unique_x = sorted(list(set([d['density'] for d in data])))
            avg_y = [np.mean([d['msd'] for d in data if d['density'] == ux]) for ux in unique_x]
            
            # Fill under curve with a soft gradient look
            fill_color = pg.mkColor(color)
            fill_color.setAlpha(25)
            plot.plot(unique_x, avg_y, pen=pg.mkPen(color, width=3), name=run_id, fillLevel=0, fillBrush=fill_color)

    def _on_scatter_clicked(self, scatter, points):
        if points:
            ds, idx = points[0].data()
            self.mazeSelected.emit(ds, idx)