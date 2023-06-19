# Import modules
import torch
from torch.utils.data import Dataset, DataLoader


# Define class
class HeightWeightDataset(Dataset):
    
    # Initialize class
    def __init__(self, csv_path):
        # Make an empty list
        self.data = []

        # Open the csv file
        with open(csv_path, 'r', encoding='utf-8') as f:
            #Skip the first line(header)
            next(f)

            # Read each line
            for line in f:
                #print(line)
                _, height, weight = line.strip().split(',')
                height = float(height)
                weight = float(weight)
                covert_to_kg_data = round(self.convert_to_kg(weight), 2)  # '2': round 2 decimal places
                convert_to_cm_data = round(self.inch_to_cm(height), 1)    # '1': round 1 decimal place
                #print(convert_to_cm_data, covert_to_kg_data)

                # Add data to list
                self.data.append([convert_to_cm_data, covert_to_kg_data])
                
    # Get single items
    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype = torch.float)
        return data

    # Get lengths
    def __len__(self):
        return len(self.data)
    
    # Define a function to convert weight to kg
    def convert_to_kg(self, weight_lb):
        return weight_lb * 0.45359237
    
    # Define a function to convert inch to cm
    def inch_to_cm(self, height_in):
        return height_in * 2.54


csv_path = './data/0619-height_weight.csv'
dataset = HeightWeightDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the data
for batch in dataloader:
    # Change to Rank2 Tensor
    x = batch[:, 0].unsqueeze(1)
    y = batch[:, 1].unsqueeze(1)

    print(x, y)