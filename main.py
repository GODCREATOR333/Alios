# main.py
import sys
import traceback
from PyQt5 import QtWidgets, QtCore
from engine import AliosWindow

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Catches any uncaught exceptions to prevent silent crashes."""
    print("\n--- ALIOS CRITICAL ERROR ---")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("----------------------------\n")
    # Optional: You could wire this to a QMessageBox to show the user the error.

def main():
    # 1. Enable High-DPI scaling for modern monitors (Retina/4K)
    # This makes pyqtgraph and text look crystal clear.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # 2. Hook the global exception handler
    sys.excepthook = global_exception_handler

    # 3. Initialize Application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ALIOS: Mechanistic Interpreter")
    app.setApplicationVersion("1.0.0")

    # 4. Boot Engine
    try:
        win = AliosWindow()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to initialize ALIOS Engine: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()