# Import modules
import sys
import os 
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QFileDialog, QTreeWidget, QTreeWidgetItem, QWidget


# Define a File Explorer class
class FileExplorer(QMainWindow):
    # Initialize the class
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle('File Explorer')
        self.resize(500, 400)

        # Set a button to open folders
        self.folder_button = QPushButton('Open a root folder')
        self.folder_button.clicked.connect(self.open_folder_dialog)

        # Set a folder tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(['File'])

        # Set layouts
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.folder_button)
        main_layout.addWidget(self.tree_widget)

        # Set a main widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        #Set a path
        self.folder_path = ''

    
    # Set Open Folder Dialog
    def open_folder_dialog(self):
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.directoryEntered.connect(self.set_folder_path)
        folder_dialog.accepted.connect(self.display_files)
        folder_dialog.exec_()


    # Set Folder Paths
    def set_folder_path(self, folder_path):
        self.folder_path = folder_path

    
    # Set display files
    def display_files(self):
        if self.folder_path:
            self.tree_widget.clear()

            # Set a root folder path
            root_item = QTreeWidgetItem(self.tree_widget, [self.folder_path])
            self.tree_widget.addTopLevelItem(root_item)

            #
            for dir_path, _, file_names in os.walk(self.folder_path):
                dir_item = QTreeWidgetItem(root_item, [os.path.basename(dir_path)])
                root_item.addChild(dir_item)

                for file_name in file_names:
                    file_item = QTreeWidgetItem(dir_item, [file_name])
                    dir_item.addChild(file_item)

                root_item.setExpanded(True)
                                                


# Run the class
if __name__ == '__main__':
# Check if instance is running
    if QApplication.instance() is not None:
        # Reuse the exiting instance
        app = QApplication.instance()
    else:
        # Run a new instance
        app = QApplication(sys.argv)  # 'sys.argv': a list containing commend-line arguments


    window = FileExplorer()
    window.show()
    sys.exit(app.exec())











