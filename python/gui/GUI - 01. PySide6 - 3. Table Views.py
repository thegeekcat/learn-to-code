# Import modules
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTableView
from PySide6.QtCore import QtMsgType
from PySide6.QtGui import QStandardItemModel
from PySide6.QtGui import QStandardItem


# Define MainWindow
class MainWindow(QMainWindow):
    # Initialize the class
    def __init__(self):
        super().__init__()
        
        # Set window title
        self.setWindowTitle('Table View Example')

        # Set Table View
        table_view = QTableView(self)
        self.setCentralWidget(table_view)  # Set 'table view' as the Central Widget

        # Create models
        model = QStandardItemModel(4, 3, self)  # Create a table with 4rows x 3columns
        model.setHorizontalHeaderLabels(['Name', 'Age', 'Sex'])  # Set header labels

        # Add data
        model.setItem(0, 0, QStandardItem('Alice'))
        model.setItem(0, 1, QStandardItem('25'))
        model.setItem(0, 2, QStandardItem('Female'))

        model.setItem(1, 0, QStandardItem('Kanen'))
        model.setItem(1, 1, QStandardItem('35'))
        model.setItem(1, 2, QStandardItem('Female'))

        model.setItem(2, 0, QStandardItem('Bob'))
        model.setItem(2, 1, QStandardItem('28'))
        model.setItem(2, 2, QStandardItem('Male'))

        model.setItem(3, 0, QStandardItem('Chris'))
        model.setItem(3, 1, QStandardItem('15'))
        model.setItem(3, 2, QStandardItem('Male'))


        # Set table view
        table_view.setModel(model)
        table_view.resizeColumnsToContents()
        table_view.setEditTriggers(QTableView.NoEditTriggers)  # 'NoEditTriggers': No Edit allowed

# Run main
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