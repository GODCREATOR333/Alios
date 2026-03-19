import pyqtgraph as pg
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal


class MazeView(pg.PlotWidget):

    cellClicked = QtCore.pyqtSignal(int, int, int, int) # r, c, state_id , aliased_count
        

    def __init__(self, title=""):
        super().__init__()
        # --- Standard Aesthetic Setup ---
        self.setBackground('#121212') 
        self.default_title = title
        self.setTitle(self.default_title, color='#AAAAAA', size='14pt')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.getViewBox().invertY(True)

        # --- Lock the view range so it stops resizing ---
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.getViewBox().disableAutoRange()
        self.setRange(xRange=[-0.5, 15.5], yRange=[-0.5, 15.5], padding=0)
        
        self.img = pg.ImageItem()
        self.img.setPos(-0.5, -0.5) 
        self.addItem(self.img)

        # THE HIGHLIGHT MASK (One single item for all highlights)
        self.highlight_mask = pg.ImageItem()
        self.highlight_mask.setPos(-0.5, -0.5)
        self.highlight_mask.setZValue(10) # High Z-index to be on top
        self.addItem(self.highlight_mask)

        # Trajectory Line (Dynamical Probe)
        # Bright Magenta, dashed line, thick width
        self.trajectory_line = pg.PlotCurveItem(
            pen=pg.mkPen(color='#FF00FF', width=4, style=QtCore.Qt.DashLine)
        )
        self.trajectory_line.setZValue(11) # Highest Z-value so it sits on top of everything
        self.addItem(self.trajectory_line)

        # Markers
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

        self.s_text = pg.TextItem("S", color='w', anchor=(0.5, 0.5))
        self.s_text.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        self.s_text.setPos(0, 0)
        self.s_text.setZValue(2)
        self.addItem(self.s_text)
        
        self.g_text = pg.TextItem("G", color='w', anchor=(0.5, 0.5))
        self.g_text.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        self.g_text.setPos(15, 15)
        self.g_text.setZValue(2)
        self.addItem(self.g_text)

        # Data storage for logic
        self.maze_data = None 
        self.current_state_map = None
        self.arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

        self.scene().sigMouseClicked.connect(self.on_mouse_click)

        # --- NEW: Heatmap Legend ---

        self.legend_img = pg.ImageItem()
        self.addItem(self.legend_img)
        self.legend_img.setVisible(False)

        self.legend_min_txt = pg.TextItem("", color='w', anchor=(0, 0.5))
        self.legend_min_txt.setPos(-1, 16.45)
        self.addItem(self.legend_min_txt)

        self.legend_max_txt = pg.TextItem("0 (Goal)", color='w', anchor=(1, 0.5))
        self.legend_max_txt.setPos(16.5, 16.45)
        self.addItem(self.legend_max_txt)


        # Draw Grid Lines
        grid_pen = pg.mkPen(color='#333333', width=1)
        for i in range(17):
            v_line = QtWidgets.QGraphicsLineItem(i - 0.5, -0.5, i - 0.5, 15.5)
            v_line.setPen(grid_pen)
            v_line.setZValue(3)
            self.addItem(v_line)
            
            h_line = QtWidgets.QGraphicsLineItem(-0.5, i - 0.5, 15.5, i - 0.5)
            h_line.setPen(grid_pen)
            h_line.setZValue(3)
            self.addItem(h_line)

        # Object Pooling for Arrows
        self.arrow_pool =[]
        self.arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arrow_font = QtGui.QFont('Arial', 10, QtGui.QFont.Bold)

        for r in range(16):
            row_items =[]
            for c in range(16):
                txt = pg.TextItem("", anchor=(0.5, 0.5))
                txt.setFont(arrow_font)
                txt.setPos(c, r)
                txt.setZValue(4)
                self.addItem(txt)
                row_items.append(txt)
            self.arrow_pool.append(row_items)

        # --- NEW: PROBE STATE VARIABLES ---
        self.current_state_map = None
        self.highlight_boxes =[]
        
        # Connect Mouse Click Event
        self.scene().sigMouseClicked.connect(self.on_mouse_click)


    def set_maze(self, maze_data):
        self.maze_data = maze_data
        h, w = maze_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[maze_data == 1] = [30, 30, 30]
        rgb[maze_data == 0] = [230, 230, 230]
        self.img.setImage(np.transpose(rgb, (1, 0, 2)))
        self.clear_highlights()
        self.clear_trajectory()
        
        # Hide legend on normal maze views
        self.legend_img.setVisible(False)
        self.legend_min_txt.setVisible(False)
        self.legend_max_txt.setVisible(False)

    def draw_trajectory(self, path):
        """path: list of (r, c) tuples"""
        if not path:
            self.clear_trajectory()
            return
            
        # PyQtGraph plots (x, y) which is (col, row)
        x = [c for r, c in path]
        y = [r for r, c in path]
        self.trajectory_line.setData(x, y)

    def clear_trajectory(self):
        self.trajectory_line.setData([],[])

    def draw_policy_vectorized(self, maze, action_grid, oracle_actions=None, base_color='#00BFFF'):
        for r in range(16):
            for c in range(16):
                item = self.arrow_pool[r][c]
                
                if maze[r, c] == 1 or (r == 15 and c == 15):
                    new_text = ""
                    new_color = base_color
                else:
                    action = int(action_grid[r, c])
                    new_text = self.arrow_map[action]
                    
                    new_color = base_color
                    if oracle_actions is not None:
                        if action != int(oracle_actions[r, c]):
                            new_color = '#FF4500' # Red

                if item.toPlainText() != new_text:
                    item.setText(new_text)
                
                if new_text != "":
                    item.setColor(pg.mkColor(new_color))

    # --- NEW: ALIASING PROBE LOGIC ---

    def set_state_map(self, state_map):
        """Stores the 16x16 grid of State IDs so we can look them up on click."""
        self.current_state_map = np.array(state_map)

    def clear_highlights(self):
        # Create a completely transparent 16x16x4 (RGBA) image
        blank = np.zeros((16, 16, 4), dtype=np.uint8)
        self.highlight_mask.setImage(blank)
        self.setTitle(self.default_title, color='#AAAAAA', size='14pt')

    def on_mouse_click(self, event):
        """Fires when the user clicks the maze."""
        if event.button() == QtCore.Qt.LeftButton:
            pos = self.getViewBox().mapSceneToView(event.scenePos())
            
            # --- FIX: Safety Clipping to prevent IndexError ---
            c = int(np.clip(round(pos.x()), 0, 15))
            r = int(np.clip(round(pos.y()), 0, 15))

            # --- FIX: Logic to ignore walls ---
            if self.maze_data is not None and self.maze_data[r, c] == 1:
                return # Don't probe walls
            
            if self.current_state_map is not None:
                state_id = int(self.current_state_map[r, c])
                
                # 1. Highlight them (this method returns the count now)
                count = self.highlight_aliased_states(state_id)
                
                # 2. Emit signal with ALL 4 pieces of data
                self.cellClicked.emit(r, c, state_id, count)

    def highlight_aliased_states(self, state_id):
        # Create an RGBA image for the mask
        # Shape (Columns, Rows, 4)
        mask_data = np.zeros((16, 16, 4), dtype=np.uint8)
        
        # Find aliased coordinates
        aliased_coords = np.argwhere(self.current_state_map == state_id)
        
        # "Paint" the mask: Gold color [255, 215, 0] with 120 alpha (transparency)
        for (r, c) in aliased_coords:
            # We only highlight paths, not walls
            if self.maze_data[r, c] == 0:
                mask_data[c, r] = [255, 215, 0, 120] 
        
        self.highlight_mask.setImage(mask_data)
        
        # Update Title
        self.setTitle(f"{self.default_title} | <span style='color: #FFD700;'>State ID: {state_id} ({len(aliased_coords)} locations)</span>")
        return len(aliased_coords)
    

    def set_heatmap(self, maze_data, value_data):
        self.clear_highlights()
        v = np.array(value_data)
        
        # --- THE BUG FIX: Mask out -inf values so they don't break the math! ---
        is_finite = np.isfinite(v)
        valid_mask = (maze_data == 0) & is_finite
        
        if not np.any(valid_mask): return
        
        v_min = np.min(v[valid_mask])
        v_max = np.max(v[valid_mask])
        
        if v_max == v_min:
            v_max = v_min + 1e-6 

        v_norm = (v - v_min) / (v_max - v_min + 1e-6)
        v_norm = np.clip(v_norm, 0, 1)

        # Get Viridis Colormap
        lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
        
        v_indices = (v_norm * 255).astype(np.uint8)
        rgb_data = lut[v_indices] 

        rgb_data[maze_data == 1] = [18, 18, 18] 
        
        # --- NEW: Color Unreachable paths Dark Red ---
        rgb_data[(maze_data == 0) & ~is_finite] = [80, 0, 0] 

        self.img.setImage(np.transpose(rgb_data, (1, 0, 2)))
        
        # Hide arrows
        for r in range(16):
            for c in range(16):
                self.arrow_pool[r][c].setText("")

        # --- DRAW HORIZONTAL LEGEND ---
        grad_indices = np.arange(256, dtype=np.uint8)
        grad_rgb = lut[grad_indices]
        grad_img = np.tile(grad_rgb[:, np.newaxis, :], (1, 4, 1))

        self.legend_img.setImage(grad_img)

        maze_width = maze_data.shape[1]
        legend_ratio = 0.7
        legend_width = maze_width * legend_ratio
        x_start = (maze_width - legend_width) / 2

        self.legend_img.setRect(x_start, 16.25, legend_width, 0.4)

        self.legend_img.setImage(grad_img)
        self.legend_min_txt.setText(f"Low: {v_min:.0f}")
        self.legend_max_txt.setText(f"High: {v_max:.0f}")

        self.legend_min_txt.setPos(x_start, 17)
        self.legend_max_txt.setPos(x_start + legend_width, 17)

        self.legend_img.setVisible(True)
        self.legend_min_txt.setVisible(True)
        self.legend_max_txt.setVisible(True)