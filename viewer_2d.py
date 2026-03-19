import pyqtgraph as pg
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class MazeView(pg.PlotWidget):
    def __init__(self, title=""):
        super().__init__()
        
        # --- Standard Aesthetic Setup (Same as before) ---
        self.setBackground('#121212') 
        self.setTitle(title, color='#AAAAAA', size='14pt')
        self.setAspectLocked(True)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.getViewBox().invertY(True)
        
        self.img = pg.ImageItem()
        self.img.setPos(-0.5, -0.5) 
        self.addItem(self.img)

        self.start_marker = QtWidgets.QGraphicsRectItem(-0.5, -0.5, 1, 1)
        self.start_marker.setBrush(QtGui.QBrush(QtGui.QColor('#0078D7'))) 
        self.start_marker.setPen(pg.mkPen(None))
        self.addItem(self.start_marker)
        
        self.goal_marker = QtWidgets.QGraphicsRectItem(14.5, 14.5, 1, 1)
        self.goal_marker.setBrush(QtGui.QBrush(QtGui.QColor('#28A745'))) 
        self.goal_marker.setPen(pg.mkPen(None))
        self.addItem(self.goal_marker)

        self.s_text = pg.TextItem("S", color='w', anchor=(0.5, 0.5))
        self.s_text.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        self.s_text.setPos(0, 0)
        self.addItem(self.s_text)
        
        self.g_text = pg.TextItem("G", color='w', anchor=(0.5, 0.5))
        self.g_text.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold))
        self.g_text.setPos(15, 15)
        self.addItem(self.g_text)

        # Draw Grid Lines
        grid_pen = pg.mkPen(color='#333333', width=1)
        for i in range(17):
            # Vertical line
            v_line = QtWidgets.QGraphicsLineItem(i - 0.5, -0.5, i - 0.5, 15.5)
            v_line.setPen(grid_pen)
            self.addItem(v_line)

            # Horizontal line
            h_line = QtWidgets.QGraphicsLineItem(-0.5, i - 0.5, 15.5, i - 0.5)
            h_line.setPen(grid_pen)
            self.addItem(h_line)

        # --- NEW: OBJECT POOLING ---
        # Instead of a list, we make a 2D array of TextItems once.
        self.arrow_pool = []
        self.arrow_map = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arrow_font = QtGui.QFont('Arial', 10, QtGui.QFont.Bold)

        for r in range(16):
            row_items = []
            for c in range(16):
                txt = pg.TextItem("", anchor=(0.5, 0.5))
                txt.setFont(arrow_font)
                txt.setPos(c, r)
                txt.setZValue(4)
                self.addItem(txt)
                row_items.append(txt)
            self.arrow_pool.append(row_items)

    def set_maze(self, maze_data):
        h, w = maze_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[maze_data == 1] = [30, 30, 30]
        rgb[maze_data == 0] = [230, 230, 230]
        self.img.setImage(np.transpose(rgb, (1, 0, 2)))

    def draw_policy(self, maze, policy_matrix, decoder_fn, oracle_policy=None, base_color='#00BFFF'):
        """Updates the text and color of the existing arrow pool."""
        
        # No more self.clear_policy()!
        
        for r in range(16):
            for c in range(16):
                # Get the existing item from the pool
                item = self.arrow_pool[r][c]
                
                # If Wall or Goal, hide the arrow
                if maze[r, c] == 1 or (r == 15 and c == 15):
                    item.setText("")
                    continue
                
                # --- IDENTICAL LOGIC TO YOUR WORKING VERSION ---
                if policy_matrix.shape == (16, 16):
                    action = policy_matrix[r, c]
                elif policy_matrix.shape[-1] == 4:
                    state_id = decoder_fn(maze, r, c)
                    action = np.argmax(policy_matrix[state_id])
                else:
                    action = 0
                
                arrow_color = base_color
                if oracle_policy is not None:
                    oracle_action = oracle_policy[r, c]
                    if int(action) != int(oracle_action):
                        arrow_color = '#FF4500' # RED
                
                # --- UPDATE EXISTING OBJECT PROPERTIES ---
                item.setText(self.arrow_map[int(action)])
                item.setColor(arrow_color)


    def draw_policy_vectorized(self, maze, action_grid, oracle_actions=None, base_color='#00BFFF'):
        """
        Updates arrow pool using pre-calculated 16x16 grids.
        action_grid: (16, 16) array of action indices (0-3)
        """
        for r in range(16):
            for c in range(16):
                item = self.arrow_pool[r][c]
                
                # Hide arrow if it's a wall or the Goal
                if maze[r, c] == 1 or (r == 15 and c == 15):
                    item.setText("")
                    continue
                
                action = action_grid[r, c]
                
                # Check Divergence
                color = base_color
                if oracle_actions is not None:
                    # Both are now 16x16 grids, so comparison is direct
                    if int(action) != int(oracle_actions[r, c]):
                        color = '#FF4500' # Red

                # Update the object in the pool
                item.setText(self.arrow_map[int(action)])
                item.setColor(color)