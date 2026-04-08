import pyqtgraph as pg
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class MazeView(pg.PlotWidget):
    cellClicked = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, title=""):
        super().__init__()
        self._setup_canvas(title)
        self._init_layers()
        self._init_grid()
        self._init_arrow_pool()
        self._init_legend()
        
        self.scene().sigMouseClicked.connect(self.on_mouse_click)
        self.last_clicked_pos = None

    def _setup_canvas(self, title):
        """Standardizes the look of the plot."""
        self.setBackground('#121212') 
        self.default_title = title
        self.setTitle(title, color='#AAAAAA', size='12pt')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.getViewBox().invertY(True)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.setRange(xRange=[-0.5, 15.5], yRange=[-0.5, 15.5], padding=0)

    def _init_layers(self):
        """Creates the Photoshop-style layers."""
        # 1. Base Image (Maze / Heatmap)
        self.img = pg.ImageItem()
        self.img.setPos(-0.5, -0.5)
        self.addItem(self.img)

        # 2. Highlight Layer (Aliasing boxes)
        self.highlight_mask = pg.ImageItem()
        self.highlight_mask.setPos(-0.5, -0.5)
        self.highlight_mask.setZValue(10)
        self.addItem(self.highlight_mask)

        # 3. Trajectory Layer
        self.trajectory_line = pg.PlotCurveItem(
            pen=pg.mkPen(color='#FF00FF', width=4, style=QtCore.Qt.DashLine)
        )
        self.trajectory_line.setZValue(11)
        self.addItem(self.trajectory_line)

        # Add Start/Goal Markers
        self.start_marker = QtWidgets.QGraphicsRectItem(-0.5, -0.5, 1, 1)
        self.start_marker.setBrush(QtGui.QBrush(QtGui.QColor('#0078D7'))) 
        self.start_marker.setPen(pg.mkPen(None))
        self.start_marker.setZValue(1)
        self.addItem(self.start_marker)
        
        self.goal_marker = QtWidgets.QGraphicsRectItem(14.5, 14.5, 1, 1)
        self.goal_marker.setBrush(QtGui.QBrush(QtGui.QColor('#28A745'))) 
        self.goal_marker.setPen(pg.mkPen(None))
        self.goal_marker.setZValue(1)
        self.addItem(self.goal_marker)

        # Add "S" and "G" text
        for label, pos in [("S", (0, 0)), ("G", (15, 15))]:
            txt = pg.TextItem(label, color='w', anchor=(0.5, 0.5))
            txt.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
            txt.setPos(pos[1], pos[0])
            txt.setZValue(2)
            self.addItem(txt)

    def _init_grid(self):
        """Draws the 16x16 grid lines."""
        grid_pen = pg.mkPen(color='#333333', width=1)
        # We need 17 lines to close the 16 boxes
        for i in range(17):
            # Vertical lines
            v_line = QtWidgets.QGraphicsLineItem(i - 0.5, -0.5, i - 0.5, 15.5)
            v_line.setPen(grid_pen)
            v_line.setZValue(3) # Place above image (0) but below arrows (5)
            self.addItem(v_line)
            
            # Horizontal lines
            h_line = QtWidgets.QGraphicsLineItem(-0.5, i - 0.5, 15.5, i - 0.5)
            h_line.setPen(grid_pen)
            h_line.setZValue(3)
            self.addItem(h_line)

    def _init_arrow_pool(self):
        """Creates 256 pre-positioned text items to avoid lag."""
        self.arrow_pool = []
        font = QtGui.QFont('Arial', 10, QtGui.QFont.Bold)
        for r in range(16):
            row = []
            for c in range(16):
                txt = pg.TextItem("", anchor=(0.5, 0.5))
                txt.setFont(font)
                txt.setPos(c, r)
                txt.setZValue(5)
                self.addItem(txt)
                row.append(txt)
            self.arrow_pool.append(row)
        self.arrow_chars = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def _init_legend(self):
        """Helper for the bottom heatmap scale."""
        self.legend_img = pg.ImageItem()
        self.legend_img.setVisible(False)
        self.addItem(self.legend_img)
        
        self.legend_min_txt = pg.TextItem("", color='#888', anchor=(0, 0))
        self.legend_max_txt = pg.TextItem("", color='#888', anchor=(1, 0))
        self.addItem(self.legend_min_txt)
        self.addItem(self.legend_max_txt)

    # --- API METHODS (Used by Engine) ---

    def set_maze(self, maze_data):
        """Renders a standard black/white maze."""
        self.clear_visuals()
        rgb = np.full((16, 16, 3), 230, dtype=np.uint8) # Default white
        rgb[maze_data == 1] = [30, 30, 30]             # Walls are dark
        self.img.setImage(np.transpose(rgb, (1, 0, 2)))

    def draw_policy(self, maze, action_grid, comparison_grid=None, color='#00BFFF'):
        """Updates the arrow pool based on policy."""
        
        # --- ADD THIS SAFETY CHECK ---
        if action_grid is None:
            return
        # -----------------------------

        for r in range(16):
            for c in range(16):
                item = self.arrow_pool[r][c]
                if maze[r,c] == 1 or (r==15 and c==15):
                    item.setText("")
                    continue
                
                action = int(action_grid[r, c])
                item.setText(self.arrow_chars[action])
                
                # If we have a comparison baseline and they disagree, turn red!
                if comparison_grid is not None and action != int(comparison_grid[r,c]):
                    item.setColor('#FF4500')
                else:
                    item.setColor(color)

    def set_heatmap(self, maze_data, values, cmap='viridis'):
        """Renders values (V or H) as a colorful heatmap."""
        self.clear_visuals(keep_image=True)
        
        # Filter out infinities (common in RL)
        valid = (maze_data == 0) & np.isfinite(values)
        if not np.any(valid): return

        v_min, v_max = values[valid].min(), values[valid].max()
        norm = (values - v_min) / (v_max - v_min + 1e-6)
        
        lut = pg.colormap.get(cmap).getLookupTable(0.0, 1.0, 256)
        rgb = lut[(np.clip(norm, 0, 1) * 255).astype(np.uint8)]
        
        rgb[maze_data == 1] = [18, 18, 18]        # Paint walls dark
        rgb[(maze_data == 0) & ~np.isfinite(values)] = [80, 0, 0] # Unreachable = Deep Red
        
        self.img.setImage(np.transpose(rgb, (1, 0, 2)))
        self._update_legend(v_min, v_max, lut)

    def highlight_aliased_states(self, state_id, state_map, maze_data, color_rgba):
        """Paints specific states (Neuro-Probe)."""
        mask = np.zeros((16, 16, 4), dtype=np.uint8)
        coords = np.argwhere(state_map == state_id)
        for (r, c) in coords:
            if maze_data[r, c] == 0:
                mask[c, r] = color_rgba # Note: PyQtGraph uses (x, y) coordinates
        self.highlight_mask.setImage(mask)

    def draw_trajectory(self, path):
        if not path or len(path) < 2:
            self.trajectory_line.setData([], [])
            return
            
        # Convert list of (r, c) to two separate 1D numpy arrays
        # x is columns (c), y is rows (r)
        y_coords = np.array([p[0] for p in path], dtype=float)
        x_coords = np.array([p[1] for p in path], dtype=float)
        
        self.trajectory_line.setData(x_coords, y_coords)

    def clear_visuals(self, keep_image=False):
        """Resets the view layers."""
        if not keep_image: 
            self.img.clear()
        self.highlight_mask.clear()
        self.trajectory_line.setData([], [])
        self.legend_img.setVisible(False)
        self.legend_min_txt.setText("")
        self.legend_max_txt.setText("")
        for row in self.arrow_pool:
            for item in row: item.setText("")

    def _update_legend(self, v_min, v_max, lut):
        grad = np.tile(lut[:, np.newaxis, :], (1, 4, 1))
        self.legend_img.setImage(grad)
        self.legend_img.setRect(2, 16.2, 12, 0.4)
        self.legend_img.setVisible(True)
        self.legend_min_txt.setText(f"{v_min:.1f}")
        self.legend_max_txt.setText(f"{v_max:.1f}")
        self.legend_min_txt.setPos(2, 16.6)
        self.legend_max_txt.setPos(14, 16.6)

    def on_mouse_click(self, event):
        pos = self.getViewBox().mapSceneToView(event.scenePos())
        c, r = int(np.floor(pos.x() + 0.5)), int(np.floor(pos.y() + 0.5))
        if 0 <= r < 16 and 0 <= c < 16:
            if (r, c) == self.last_clicked_pos:
                self.clear_visuals(keep_image=True)
                self.cellClicked.emit(-1, -1, -1, 0)
                self.last_clicked_pos = None
            else:
                self.last_clicked_pos = (r, c)
                self.cellClicked.emit(r, c, 0, 0)