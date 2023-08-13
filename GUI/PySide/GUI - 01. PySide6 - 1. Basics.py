# Import modules
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton


# Define a class for main window
class MainWindow(QMainWindow):
    # Initialize the class
    def __init__(self):
        super().__init__()
        
        # Set a title
        self.setWindowTitle('Example of Resizing Window')

        # Set a size
        self.resize(500, 500)

        # Set a button
        self.button = QPushButton('Click', self)
        self.button.setGeometry(50, 50, 400, 50)
    
    # Set clicked button
    def buttonClicked(self):
        print('Button clicked')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())
