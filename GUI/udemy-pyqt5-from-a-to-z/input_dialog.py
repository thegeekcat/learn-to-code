import sys
from PyQt5.QtWidgets import *


class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My GUI')
        self.resize(200, 200)

        self.btn = QPushButton('Name', self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.evt_btn_clicked_name)

        self.btn = QPushButton('Age', self)
        self.btn.move(20, 50)
        self.btn.clicked.connect(self.evt_btn_clicked_age)

        self.btn = QPushButton('Coffee Cost', self)
        self.btn.move(20, 80)
        self.btn.clicked.connect(self.evt_btn_clicked_cost)

        self.btn = QPushButton('Favourite Color', self)
        self.btn.move(20, 110)
        self.btn.clicked.connect(self.evt_btn_clicked_color)

    # Button: Name
    def evt_btn_clicked_name(self):
        sName, bOk = QInputDialog.getText(self, 'Text', 'Enter your name: ')
        if bOk:
            QMessageBox.information(self, 'Name', 'Your name is ' + sName)
        else:
            QMessageBox.critical(self, 'Canceled', 'User canceled')

    # Button: Age
    def evt_btn_clicked_age(self):
        iAge, bOk = QInputDialog.getInt(self, 
                                          'Text', 
                                          'Enter your age: ', 
                                          18,  # Default value
                                          18,  # Minimum
                                          65,  # Maximum
                                          1)   # Number of digits after the decimall point
        if bOk:
            QMessageBox.information(self, 'Name', 'Your age is ' + str(iAge) + ' years old')
        else:
            QMessageBox.critical(self, 'Canceled', 'User canceled')

    # Button: Cost
    def evt_btn_clicked_cost(self):
        dCost, bOk = QInputDialog.getDouble(self, 
                                          'Text', 
                                          'Enter the cost of your last coffee:', 
                                          2.00,  # Default value
                                          1.50,  # Minimum
                                          10.00,  # Maximum
                                          2)   # Number of digits after the decimall point
        if bOk:
            QMessageBox.information(self, 'Name', 'Your coffee cost was ' + str(dCost))
        else:
            QMessageBox.critical(self, 'Canceled', 'User canceled')


    # Button: Color
    def evt_btn_clicked_color(self):
        lstColor = ['Red', 'Green', 'Blue', 'Yellow', 'Orange']
        sColor, bOk = QInputDialog.getItem(self, 'Text', 'Enter your favourite color: ', lstColor)
        if bOk:
            QMessageBox.information(self, 'Name', 'Your favourite color is ' + sColor + '!')
        else:
            QMessageBox.critical(self, 'Canceled', 'User canceled')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
