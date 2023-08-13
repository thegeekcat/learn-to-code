import sys
from PyQt5.QtWidgets import *


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My GUI')
        self.resize(200, 200)

        self.btn = QPushButton('Show Message', self)
        self.btn.move(40, 40)
        self.btn.clicked.connect(self.evt_btn_clicked)

    def evt_btn_clicked(self):
        msgDiskFull = QMessageBox()

        # Messsage box with a message:
        msgDiskFull.setText('Your hard drive is almost full! :( ')

        # Message when 'Show Details' button is clicked
        msgDiskFull.setDetailedText('Please make some room on your disk!')

        # Icon
        msgDiskFull.setIcon(QMessageBox.Information)

        # Window title
        msgDiskFull.setWindowTitle('Full Drive')

        # Buttons
        msgDiskFull.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # Quit
        msgDiskFull.exec_()






        # res = QMessageBox.critical(self, 'Disk Full', 'Your disk drive is almost full :(')
        #                # Options: information, question, warning, critical
        # if res == QMessageBox.Yes:
        #     QMessageBox.information(self, '', 'You clicked Yes!')
        # else:
        #     QMessageBox.information(self, '', 'You clicked No!')

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
