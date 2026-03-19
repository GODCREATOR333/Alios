import sys
from PyQt5 import QtWidgets
from engine import AliosWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = AliosWindow()
    win.show()
    sys.exit(app.exec_())
    