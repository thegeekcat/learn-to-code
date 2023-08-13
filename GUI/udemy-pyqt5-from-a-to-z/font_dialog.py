import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *  # for 'QFont'


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My GUI')
        self.resize(250, 200)

        self.btn = QPushButton('Choose Font', self)
        self.btn.move(20, 50)     # Position of the button
        self.btn.resize(200, 50)  # Resize the button
        font = QFont('Arial', 20, 75, True)   # Initialize font
        self.btn.setFont(font)                # Initialize the font
        self.btn.clicked.connect(self.evt_btn_clicked)



    # Button: Choose Color
    def evt_btn_clicked(self):
        font, bOk = QFontDialog.getFont()
        print(font, bOk)
        if bOk:
            print('Font Family: ', font.family())
            print('Italic: ', font.italic())
            print('Bold: ', font.bold())
            print('Weight: ', font.weight())
            print('Font Size: ', font.pointSize())
            self.btn.setFont(font)   # Set the font of the button to the chosen font




if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
