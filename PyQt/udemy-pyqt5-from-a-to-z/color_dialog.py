import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *  # for 'QColor'


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My GUI')
        self.resize(200, 200)

        self.btn = QPushButton('Choose Color', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.evt_btn_clicked)



    # Button: Choose Color
    def evt_btn_clicked(self):
        color = QColorDialog.getColor(QColor('#663300'),  # Default color
                                      self,               # Parent widge
                                      'Choose color')     # Window title
        print(color)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
