import pyqtgraph as pg
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class MazeView(pg.PlotWidget):
    def __init__(self, title=""):
        super().__init__()
        
        # --- 1. Aesthetic Styles ---
        self.setBackground('#121212') 
        self.setTitle(title, color='#AAAAAA', size='14pt')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.getViewBox().invertY(True)
        
        # --- 2. Layers (Z-Value is important) ---
        # Image is at the bottom (Z=0)
        self.img = pg.ImageItem()
        # FIX: Shift image by half a cell to align with grid lines
        self.img.setPos(-0.5, -0.5) 
        self.addItem(self.img)
        self.img.setZValue(0)

        # Markers for Start and Goal (Z=1)
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

        # Labels (Z=2)
        self.s_text = pg.TextItem("S", color='w', anchor=(0.5, 0.5))
        self.s_text.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        self.s_text.setPos(0, 0)
        self.addItem(self.s_text)

        self.g_text = pg.TextItem("G", color='w', anchor=(0.5, 0.5))
        self.g_text.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        self.g_text.setPos(15, 15)
        self.addItem(self.g_text)

        # --- 3. Crisp Grid Lines (Z=3) ---
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

        self.arrow_items = []
        self.arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def set_maze(self, maze_data):
        h, w = maze_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Walls: Dark Gray
        rgb[maze_data == 1] = [30, 30, 30]
        # Path: Very Light Gray
        rgb[maze_data == 0] = [230, 230, 230]
        # Transpose to match (X, Y) layout of PlotWidget
        self.img.setImage(np.transpose(rgb, (1, 0, 2)))

    def draw_policy(self, maze, policy_matrix, decoder_fn, oracle_policy=None, base_color='#00BFFF'):
        self.clear_policy()
        font = QtGui.QFont('Arial', 10, QtGui.QFont.Bold)

        for r in range(16):
            for c in range(16):
                if maze[r, c] == 1 or (r == 15 and c == 15):
                    continue
                
                # --- THE FIX: Differentiate between a Grid and a Q-Table ---
                if policy_matrix.shape == (16, 16):
                    # It's a spatial grid of actions (like the Oracle VI Policy)
                    action = policy_matrix[r, c]
                elif policy_matrix.shape[-1] == 4:
                    # It's a Q-table (States, Actions)
                    state_id = decoder_fn(maze, r, c)
                    action = np.argmax(policy_matrix[state_id])
                else:
                    action = 0 # Fallback safety
                
                # --- Color Logic ---
                arrow_color = base_color
                if oracle_policy is not None:
                    # Oracle policy is ALWAYS a 16x16 grid
                    oracle_action = oracle_policy[r, c]
                    if int(action) != int(oracle_action):
                        arrow_color = '#FF4500' # RED (Divergence/Sub-optimal)
                
                txt = pg.TextItem(text=self.arrow_map[int(action)], color=arrow_color, anchor=(0.5, 0.5))
                txt.setFont(font)
                txt.setPos(c, r)
                txt.setZValue(4)
                self.addItem(txt)
                self.arrow_items.append(txt)

    def clear_policy(self):
        for item in self.arrow_items:
            self.removeItem(item)
        self.arrow_items = []