# Import modules
import os
import matplotlib.pyplot as plt

# Define a class
class DataVisualizer:
    # Initialize class
    def __init__(self, directory_data):
        self.directory_data = directory_data

        # Initialize lists of data
        self.data_all = {}
        self.data_train = {}
        self.data_validation = {}
        self.data_test = {}


    # Set Load data
    def load_data(self):
        # Set directories
        directory_train = os.path.join(self.directory_data, 'Train')
        directory_validation = os.path.join(self.directory_data, 'Validation')
        directory_test = os.path.join(self.directory_data, 'Test')
        # print(directory_train)      # Result: ./data/0717-Food Dataset\Train
        # print(directory_validation) # Result: ./data/0717-Food Dataset\Validation
        # print(directory_test)       # Result: ./data/0717-Food Dataset\Test
        # exit()

        # Set labels for Train Data
        for label in os.listdir(directory_train):
            label_directory = os.path.join(directory_train, label)
            #print(label_directory)  # Result: ./data/0717-Food Dataset\Train\burger
            #exit()
            count = len(os.listdir(label_directory))
            self.data_all[label] = count
            self.data_train[label] = count


        # Set labels for Validation Data
        for label in os.listdir(directory_validation):
            label_directory = os.path.join(directory_validation, label)
            # print(label_directory)  # Result: ./data/0717-Food Dataset\Validation\burger
            # exit()
            count = len(os.listdir(label_directory))
            self.data_validation[label] = count
            if label in self.data_all:
                self.data_all[label] += count
            else:
                self.data_all[label] = count

        # Set labels for Test Data
        for label in os.listdir(directory_test):
            label_directory = os.path.join(directory_test, label)
            # print(label_directory) # Result: ./data/0717-Food Dataset\Test\burger
            # exit()
            count = len(os.listdir(label_directory))
            self.data_test[label] = count
            if label in self.data_all:
                self.data_all[label] += count
            else:
                self.data_all[label] = count

    # Set Visualize data
    def visualize_data(self):
        # Get Labels and Counts
        labels = list(self.data_all.keys())  # 'keys()': Get key values from dictionaries
        counts = list(self.data_all.values())
        #print(labels, counts)

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts)
        plt.title('Label Data Number')
        plt.xlabel('Labels')
        plt.ylabel('Number of Data')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.show()



# Run codes
if __name__ == '__main__':
    test = DataVisualizer('./data/0717-Food Dataset')
    test.load_data()
    test.visualize_data()
