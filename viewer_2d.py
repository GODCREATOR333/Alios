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
        
        self.img = pg.ImageItem()
        self.img.setPos(-0.5, -0.5) 
        self.addItem(self.img)
        self.img.setZValue(0)

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
        h, w = maze_data.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[maze_data == 1] =[30, 30, 30]
        rgb[maze_data == 0] =[230, 230, 230]
        self.img.setImage(np.transpose(rgb, (1, 0, 2)))

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
        """Removes the yellow boxes and resets the title."""
        for box in self.highlight_boxes:
            self.removeItem(box)
        self.highlight_boxes.clear()
        self.setTitle(self.default_title, color='#AAAAAA', size='14pt')

    def on_mouse_click(self, event):
        """Fires when the user clicks the maze."""
        if event.button() == QtCore.Qt.LeftButton:
            pos = self.getViewBox().mapSceneToView(event.scenePos())
            
            # --- FIX: Safety Clipping to prevent IndexError ---
            c = int(np.clip(round(pos.x()), 0, 15))
            r = int(np.clip(round(pos.y()), 0, 15))
            
            if self.current_state_map is not None:
                state_id = int(self.current_state_map[r, c])
                
                # 1. Highlight them (this method returns the count now)
                count = self.highlight_aliased_states(state_id)
                
                # 2. Emit signal with ALL 4 pieces of data
                self.cellClicked.emit(r, c, state_id, count)

    def highlight_aliased_states(self, state_id):
        self.clear_highlights()
        aliased_coords = np.argwhere(self.current_state_map == state_id)
        
        box_pen = pg.mkPen(color='#FFD700', width=3)
        for (r, c) in aliased_coords:
            rect = QtWidgets.QGraphicsRectItem(c - 0.5, r - 0.5, 1, 1)
            rect.setPen(box_pen)
            rect.setZValue(5)
            self.addItem(rect)
            self.highlight_boxes.append(rect)
        
        # Return the count so on_mouse_click can send it to the engine
        return len(aliased_coords)