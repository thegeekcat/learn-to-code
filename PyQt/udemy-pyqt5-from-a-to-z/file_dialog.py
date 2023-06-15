import sys
from PyQt5.QtWidgets import *


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My GUI')
        self.resize(200, 200)

        self.btn = QPushButton('Open File', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.evt_btn_clicked_open)

        self.btn = QPushButton('Save File', self)
        self.btn.move(20, 50)
        self.btn.clicked.connect(self.evt_btn_clicked_save)

        self.btn = QPushButton('Open Files', self)
        self.btn.move(20, 80)
        self.btn.clicked.connect(self.evt_btn_clicked_open_files)


    # Button: Open a Single File
    def evt_btn_clicked_open(self):
        res = QFileDialog.getOpenFileName(self, 
                                          'Open File',         # Title
                                          'd:\\',              # Default directory
                                          'PNG Files (*.png);;JPG Files (*.jpg)')  # Default filter for types of files
        print(res)

    # Button: Save File
    def evt_btn_clicked_save(self):
        res = QFileDialog.getSaveFileName(self, 
                                          'Open File',         # Title
                                          'd:\\',              # Default directory
                                          'PNG Files (*.png);;JPG Files (*.jpg)')  # Default filter for types of files
        print(res)

    # Button: Open Multiple Files
    def evt_btn_clicked_open_files(self):
        res = QFileDialog.getOpenFileNames(self, 
                                          'Open File',         # Title
                                          'd:\\',              # Default directory
                                          'PNG Files (*.png);;JPG Files (*.jpg)')  # Default filter for types of files
        print(res)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
