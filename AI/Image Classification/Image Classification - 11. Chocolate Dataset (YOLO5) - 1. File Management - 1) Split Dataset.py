# Import modules
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Set paths
path_folder_images = './dataset/0724-Chocolate Dataset/Images'
path_folder_annotation = './dataset/0724-Chocolate Dataset/Annotations'
path_file_csv = os.path.join(path_folder_annotation, 'annotations.csv')

# Make folders
folder_train = './outcomes/0724-Chocolate Dataset_splited/Train'
folder_evaluation = './outcomes/0724-Chocolate Dataset_splited/Validation'
os.makedirs(folder_train, exist_ok=True)
os.makedirs(folder_evaluation, exist_ok=True)

# Load csv files to dataframe
df_annotation = pd.read_csv(path_file_csv)


# Get image names
names_images = df_annotation['filename'].unique()  # 'unique()': Get unique values
#print(name_images)  # Result: 'IMG_1930.JPG' 'IMG_1931.JPG'
#print('Length of name_images: ', len(names_images))  # Result: Length of image_names:  528

# Split dataset
names_train, names_evaluation = train_test_split(names_images, test_size=0.2)
#print('Data size of Train set: ', len(names_train))   # Result: Data size of Train set:  422
#print('Data size of Evaluation set: ', len(names_evaluation))  # Result: Data size of Evaluation set:  106


# Get info of bounding boxes for Train dataset from the annotation file
annotations_train = pd.DataFrame(columns=df_annotation.columns)
for name_images in names_train:
    #print('names_image: ', names_images)
    path_images_resource_folder = os.path.join(path_folder_images, name_images)
    #print(path_images_resource_folder)
    path_images_outcome_folder = os.path.join(folder_train, name_images)
    #print(path_images_outcome_folder)  # Result: ./outcomes/0724-Chocolate Dataset/train\IMG_2489.JPG

    # Copy data
    shutil.copy(path_images_resource_folder, path_images_outcome_folder)

    # Get annotation information to csv files
    annotation = df_annotation.loc[df_annotation['filename'] == name_images].copy()
    annotation['filename'] = name_images
    #print(annotation)  # Result: 1032  IMG_2205.JPG  ...  {"":{"hello":false},"candy_type":"3_Musketeers"}
    annotations_train = annotations_train._append(annotation)  # '._append': Append for Pandas
    #print(annotations_train)  # Result: 1032  IMG_2205.JPG  ...   {"":S{"hello":false},"candy_type":"3_Musketeers"}
print('Annotations for Train Dataset has been completed')
annotations_train.to_csv(os.path.join(folder_train, 'annotations.csv'), index=False)

    # Reference
    #  - '.append': Append for List function
    #  - '._append': Append for Pandas library


# Get info of bounding boxes for Evaluation dataset from the annotation file
annotations_evaluation = pd.DataFrame(columns=df_annotation.columns)
for name_images in names_evaluation:
    #print('names_image: ', names_images)  # Result: 'IMG_2473.JPG' 'IMG_2474.JPG'
    path_images_resource_folder = os.path.join(path_folder_images, name_images)
    #print(path_images_resource_folder)  # Result: ../../data/0724-Chocolate Dataset/Images\IMG_2178.JPG
    path_images_outcome_folder = os.path.join(folder_evaluation, name_images)
    #print(path_images_outcome_folder)  # Result: ../../data/0724-Chocolate Dataset/Annotations\IMG_2132.JPG

    # Copy data
    shutil.copy(path_images_resource_folder, path_images_outcome_folder)

    # Get annotation information to csv files
    annotation = df_annotation.loc[df_annotation['filename'] == name_images].copy()
    annotation['filename'] = name_images
    #print(annotation)  # Result: 600  IMG_2092.JPG  ...  {"":{"hello":false},"candy_type":"Butterfingers"}
    annotations_evaluation = annotations_evaluation._append(annotation)  # '._append': Append for Pandas
    #print(annotations_evaluation)  # Result: 1512  IMG_2332.JPG  ...   {"":{"hello":false},"candy_type":"3_Musketeers"}
print('Annotations for Evaluation Dataset has been completed')
annotations_evaluation.to_csv(os.path.join(folder_evaluation, 'annotations.csv'), index=False)




