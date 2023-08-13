# Import modules
import sys
from PySide6.QtWidgets import QApplication, QLabel,QVBoxLayout, QLineEdit,QPushButton,QCheckBox,QMessageBox, QWidget


# Define a class
class MainWindow(QWidget):
    # Initialize the class
    def __init__(self):
        super().__init__()

        # Set title
        self.setWindowTitle('Widget Practice')

        # Set Interfaces
        self.label = QLabel('Please input your location.')
        self.line_edit = QLineEdit()
        self.button = QPushButton('Submit')
        self.checkbox = QCheckBox('Agree to submit.')

        # Set layouts
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.button)

        self.setLayout(layout)

        # Set a clicking event
        self.button.clicked.connect(self.show_message)

    
    # Define Result part
    def show_message(self):
        # Set Accept button
        if self.checkbox.isChecked():
            message = self.line_edit.text()
            print('Your location is: ', message)
            self.line_edit.clear()
        else:
            error_message = 'Please click the agree button.'
            QMessageBox.critical(self, 'Error', error_message)
            self.line_edit.clear()

# Run the main program
if __name__ == '__main__':
    # Check if instance is running
    if QApplication.instance() is not None:
        # Reuse the exiting instance
        app = QApplication.instance()
    else:
        # Run a new instance
        app = QApplication(sys.argv)  # 'sys.argv': a list containing commend-line arguments
    
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
            


