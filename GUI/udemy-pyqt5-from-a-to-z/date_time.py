import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *  # for 'QFont'
from PyQt5.QtCore import *  # for 'QTimer'



class DlgMain(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('My GUI')
        self.resize(250, 200)

        self.btn = QPushButton('Dates', self)
        self.btn.move(20, 50)     # Position of the button
        self.btn.resize(120, 50)  # Resize the button
        font = QFont('Arial', 10, 75, True)   # Initialize font
        self.btn.setFont(font)                # Initialize the font
        self.btn.clicked.connect(self.evt_btn_clicked)



    # Button: 
    def evt_btn_clicked(self):
        dt = QDate.currentDate()
        print('to String: ', dt.toString())
        print('to Julian Day: ', dt.toJulianDay())
        print('Day of Year: ', dt.dayOfYear())
        print('Day of Week: ', dt.dayOfWeek())
        print('Add Days: ', dt.addDays(23).toString())

        tm = QTime(14, 30, 15) # Hour, Minute, Second
        print(tm.toString())

        tm2 = QTime(20, 15)    # Hour, Minute
        print(tm2.toString())
        print(tm.secsTo(tm2))  # Change time to seconds

        dtm = QDateTime.currentDateTime() # Current date and time
        print(dtm.toString())







if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlgMain = DlgMain()
    dlgMain.show()
    sys.exit(app.exec_())
