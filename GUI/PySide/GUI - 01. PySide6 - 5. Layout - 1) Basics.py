# Import modules
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLineEdit, QLabel


# Define a Main Window class
class MainWindow(QWidget):
    # Initialize the class
    def __init__(self):
        super().__init__()

        # Set window title
        self.setWindowTitle('Complex UI Example')

        ###### Group Box 1
        # Add features
        group_box1 = QGroupBox('Personal Information')
        label1 = QLabel('Name: ')
        line_edit1 = QLineEdit()
        button_save = QPushButton('Save')

        # Add layouts
        layout1 = QVBoxLayout()
        layout1.addWidget(label1)
        layout1.addWidget(line_edit1)
        layout1.addWidget(button_save)
        group_box1.setLayout(layout1)

        ##### Group Box 2
        # Add features
        group_box2 = QGroupBox('Geographic Information')
        label2 = QLabel('Location: ')
        line_edit2 = QLineEdit()
        button_search = QPushButton('Search')

        # Add layouts
        layout2 = QHBoxLayout()
        layout2.addWidget(label2)
        layout2.addWidget(line_edit2)
        layout2.addWidget(button_search)
        group_box2.setLayout(layout2)

        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)

        self.setLayout(main_layout)


# Run the class
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








