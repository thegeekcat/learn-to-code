# Import modules
import sys
import csv
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QListWidget, QMessageBox


# Define a class
class MainWindow(QWidget):
    # Initialize the class
    def __init__(self):
        ##### Common Area
        super().__init__()

        # Set window properties
        self.setWindowTitle('Student Management System')
        self.resize(500, 250)

        # Set Group Boxes
        group_box1 = QGroupBox('Information')
        group_box2 = QGroupBox('Display Input Information')
        group_box3 = QGroupBox('Save and Load Information')



        ##### Group Box 1: Information
        # Set Labels
        self.label_id = QLabel('ID')
        self.label_name = QLabel('Name')
        self.label_age = QLabel('Age')
        self.label_gender = QLabel('Sex')
        self.label_country = QLabel('Country')

        # Set Input Line Editors
        self.line_edit_id = QLineEdit()
        self.line_edit_name = QLineEdit()
        self.line_edit_age = QLineEdit()
        self.line_edit_gender = QLineEdit()
        self.line_edit_country = QLineEdit()

        # Add Widgets
        layout1 = QVBoxLayout()
        layout1.addWidget(self.label_id)
        layout1.addWidget(self.line_edit_id)
        layout1.addWidget(self.label_name)
        layout1.addWidget(self.line_edit_name)
        layout1.addWidget(self.label_age)
        layout1.addWidget(self.line_edit_age)
        layout1.addWidget(self.label_gender)
        layout1.addWidget(self.line_edit_gender)
        layout1.addWidget(self.label_country)
        layout1.addWidget(self.line_edit_country)

        # Set Layouts for Group Box 1
        group_box1.setLayout(layout1)



        ##### Group Box 2: Display Input Information
        # Set Buttons
        self.button_view = QPushButton('View Information')
        self.button_view.clicked.connect(self.info_show)
        self.button_close = QPushButton('Close')
        self.button_close.clicked.connect(self.info_close)

        # Add a space line
        self.label_display_info = QLabel()

        # Add Widgets        
        layout2 = QVBoxLayout()        
        layout2.addWidget(self.label_display_info)  # Add a space line
        layout2.addWidget(self.button_view)
        layout2.addWidget(self.button_close)
        layout2.setContentsMargins(10, 10, 10, 10)  # Set margins
        
        # Set layout for Group Box 2
        group_box2.setLayout(layout2)




        ##### Group Box 3: Save and Load Information
        # Set Buttons
        self.button_save = QPushButton('Save')
        self.button_save.clicked.connect(self.info_save)
        self.button_load = QPushButton('Load')
        self.button_load.clicked.connect(self.info_load)

        # Set a list widget
        self.list_widget = QListWidget()

        # Add Widgets
        layout3 = QVBoxLayout()
        layout3.addWidget(self.button_save)
        layout3.addWidget(self.button_load)
        layout3.addWidget(self.list_widget)
        layout3.setContentsMargins(10, 10, 10, 10)  # Set margins

        # Set Layouts for Group Box 3
        group_box3.setLayout(layout3)


        ##### Set Main Area
        # Set all layouts
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)
        main_layout.addWidget(group_box3)

        # Set Layouts for Main area
        self.setLayout(main_layout)


    # Define 'info_show' layout widget
    def info_show(self):
        ##### Group Box 1
        id = self.line_edit_id.text()
        name = self.line_edit_name.text()
        age = self.line_edit_age.text()
        gender = self.line_edit_gender.text()
        country = self.line_edit_country.text()

        ##### Group Box 3
        info_text = f'ID: {id} \nName: {name} \nAge: {age} \nGender: {gender} \nCountry: {country}'
        self.label_display_info.setText(info_text)


    # Define 'info_close' layout widget
    def info_close(self):
        self.line_edit_id.clear()
        self.line_edit_name.clear()
        self.line_edit_age.clear()
        self.line_edit_gender.clear()
        self.line_edit_country.clear()
        self.label_display_info.clear()

    # Define 'info_save' layout widget
    def info_save(self):
        # Get data
        id = self.line_edit_id.text()
        name = self.line_edit_name.text()
        age = self.line_edit_age.text()
        gender = self.line_edit_gender.text()
        country = self.line_edit_country.text()

        # Get a list of all data
        data = [id, name, age, gender, country]

        # Save as a file
        try:
            with open('./data/0711-PySide6-Layout.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            QMessageBox.information(self, 'Save completed!', 'Saved')
        except Exception as e:
            QMessageBox.critical(self, 'Failed to save file!', f'There was an error while saving as a file: \n{str(e)}')


    # Define 'info_load' layout widget
    def info_load(self):
        # Clear list
        self.list_widget.clear()

        # Load a file
        try:
            with open('./data/0711-PySide6-Layout.csv', 'r') as file:
                reader = csv.reader(file)
                # Read rows
                for row in reader:
                    data_text = f'ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Gender: {row[3]}, Country: {row[4]}'
                    self.list_widget.addItem(data_text)
        except Exception as e:
            QMessageBox.critical(self, 'Load Failed!', f'Error happened while loading: \n{str(e)}')


# Run the main
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


        




