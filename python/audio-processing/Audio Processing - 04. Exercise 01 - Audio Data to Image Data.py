##### Notes #####
# 1. Set Data folders
# 2. Data Preprocessing + Augmentation
# 3. Data Processing
##################



###### Preparation #####
# Import modules
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from tqdm import tqdm
from PIL import Image

import librosa
import librosa.display



##### 1. Set Data Folders #####
def create_folders(folder_name):
    # Set Folder Paths
    dir_extracted_images = './outcomes/0622-audio-to-image/extracted_images/'  # sound files -> extracted images
    dir_preprocessed_images = './outcomes/0622-audio-to-image/preprocessed_images/'  # extracted images -> images with aumentation

    # Generate folders
    for dir_transform_types in ['MelSpectrogram', 'STFT', 'waveshow']:
        # Create folders: sound files -> extracted images
        os.makedirs(f'{dir_extracted_images}/{dir_transform_types}/{folder_name}', exist_ok=True)

        # Create folders: extracted images -> preprocessed images
        os.makedirs(f'{dir_preprocessed_images}/{dir_transform_types}/{folder_name}', exist_ok=True)



##### 2. Data Preprocessing + Augmentation #####

##### 2.1. MelSpectrogram #####
# Define a function: MelSpectrogram with Original data
def process_melspectrogram_original(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Calculate STFT
    # Reason to calculate STFT: MelSpectrogram is a way of displaying results of STFT in a clear way
    stft_original_data = librosa.stft(data_section)

    # Calculate MelSpectrogram
    melspectrogram_original = librosa.melspectrogram(S=abs(stft_original_data))  # 'abs()': Get absolute values

    # Change to DB format
    melspectrogram_db_original = librosa.amplitude_to_db(melspectrogram_original,
                                                         ref=np.max)  # 'ref=np.max': The max value in Mel Spectrogram will be used as a reference for scaling

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(melspectrogram_db_original, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()


# Define a function: MelSpectrogram with Noise added data
def process_melspectrogram_noise(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Before Noise: Calculate STFT
    stft_before_noise = librosa.stft(data_section)

    # Before Noise: Calculate MelSpectrogram
    melspectrogram_before_noise = librosa.feature.melspectrogram(S=abs(stft_before_noise))

    # Before Noise: Change to DB format
    melspectrogram_db_before_noise = librosa.amplitude_to_db(melspectrogram_before_noise, ref=np.max)


    # Add Augmentation: Noise
    noise_melspectrogram = 0.005 * np.random.randn(* melspectrogram_db_before_noise.shape)
    data_melspectrogram_after_noise = melspectrogram_db_before_noise + noise_melspectrogram


    # After Noise: Change to DB format
    melspectrogram_db_after_noise = librosa.amplitude_to_db(data_melspectrogram_after_noise, ref=np.max)

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(melspectrogram_db_after_noise, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()


# Define a function: MelSpectrogram with Stretched data
def process_melspectrogram_stretched(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Add augmentation: Stretching
    rate_melspectrogram = np.random.uniform(low=0.8, high=1.2)
    stretched_melspectrogram = librosa.effects.time_stretch(data_section, rate=rate_melspectrogram)

    # Calculate MelSpectrogram
    melspectrogram_stretch = librosa.feature.melspectrogram(S=abs(stretched_melspectrogram))

    # Change to DB format
    melspectrogram_db_stretch = librosa.amplitude_to_db(melspectrogram_stretch, ref=np.max)

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(melspectrogram_db_stretch, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()



##### 2.2. STFT(Short-Time Fourier Transform) #####
# Define a function: STFT with Original data
def process_stft_original(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Calculate STFT
    stft = librosa.stft(data_section)

    # Change to DB format
    stft_db = librosa.amplitude_to_db(abs(stft))

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()


# Define a function: STFT with Noise added data
def process_stft_noise(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Add augmentation: Noise
    noise_stft = 0.005 * np.random.rand(*data_section.shape)
    data_stft_after_noise = data_section + noise_stft

    # Calculate stft
    stft_after_noise = librosa.stft(data_stft_after_noise)

    # Change to DB format
    stft_db_after_noise = librosa.amplitude_to_db(abs(stft_after_noise))

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(stft_db_after_noise, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()


# Define a function: STFT with Stretched data
def process_stft_stretched(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Add augmentation: Stretching
    rate_stft = 0.8 + np.random.random() * 0.4  # Randomly stretch time between 0.8% and 1.2%
    stretched_stft = librosa.effects.time_stretch(data_section, rate=rate_stft)

    # Calculate STFT
    stft_stretch = librosa.stft(stretched_stft)

    # Change to DB format
    stft_db_stretch = librosa.amplitude_to_db(abs(stft_stretch))

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(stft_db_stretch, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()



##### 2.3. waveshow #####
# Define a function: waveshow with Original data
def process_waveshow_original(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(data_section, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()

# Define a function: waveshow with Noise added data
def process_waveshow_noise(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Add augmentation: Noise
    noise_waveshow = 0.05 * np.random.rand(* data_section.shape)
    data_waveshow_after_noise = data_section + noise_waveshow

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(data_waveshow_after_noise, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()

# Define a function: waveshow with Stretched data
def process_waveshow_stretched(data_section, folder_name, file_name, aug_mode, mode, sr):
    # Add augmentation: Stretching
    rate_waveshow = 0.8
    stretched_waveshow = librosa.effects.time_stretch(data_section, rate=rate_waveshow)

    # Visualization
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(stretched_waveshow, sr=sr, x_axis='time', y_axis='hz')
    plt.axis('off')
    plt.savefig(f'./outcomes/0622-audio-to-image/extracted_images/{mode}/{folder_name}/{file_name}_{aug_mode}.png',
                bbox_inches = 'tight',
                pad_inches = 0)
    plt.close()



##### 3. Set Modes #####
MODES = {
    'MelSpectrogram': {
        'org': process_melspectrogram_original,
        'noise': process_melspectrogram_noise,
        'stretch': process_melspectrogram_stretched
    },
    'STFT': {
        'org': process_stft_original,
        'noise': process_stft_noise,
        'stretch': process_stft_stretched
    },
    'waveshow': {
        'org': process_waveshow_original,
        'noise': process_waveshow_noise,
        'stretch': process_waveshow_stretched
    }
}


##### 4. Run Codes #####
# Run codes 
if __name__ == '__main__':
    # Set a folder path
    path_raw_data = './data/0620sound_data'
    list_raw_data = glob.glob(os.path.join(path_raw_data, '*', '*.wav'))
    #Sprint(list_raw_data)  # Result: './data/0620sound_data\\blues\\blues.00000.wav'

    mode = 'MelSpectrogram'
    aug_mode = 'org'

    # Test loading data
    # raw_data = list_raw_data[1]
    # print('raw_data: ', raw_data) # Result: ./data/0620sound_data\blues\blues.00001.wav
    # data, sr = librosa.load(raw_data)
    # print(data, sr)
    # exit()

    for raw_data in list_raw_data:
        data, sr = librosa.load(raw_data)
        #print(data, sr)

        # Generate folders
        folder_name = raw_data.split('\\')[2]
        file_name = raw_data.split('\\')[-1]
        file_name = file_name.replace('.wav', '')  # Remove the extensions/
        print('Folder Name: ', folder_name)
        print('File Name: ', file_name)

        # Extract a part between 0 sec and 10 secs
        start_time = 0
        end_time = 10
        sample_start = sr * start_time
        sample_end = sr * end_time
        data_section = data[sample_start : sample_end]

        # Set values
        if mode in MODES and aug_mode in MODES[mode]:
            MODES[mode][aug_mode](data_section, folder_name, file_name, aug_mode, mode, sr)



    # # Set lists for modes and augmentation modes
    # list_mode = ['MelSpectrogram', 'STFT', 'Waveshow']
    # list_aug_mode = ['Original', 'Noise', 'Stretch']
    #
    #
    # # Save files
    # for mode in list_mode:
    #     # Set Augmentation modes
    #     #print('mode')
    #     for aug_mode in  list_aug_mode:
    #         # Set each file
    #         #print('aug_mode')
    #         for raw_data in list_raw_data:
    #             print('Raw Data: ', raw_data)
    #             data, sr = librosa.load(raw_data)
    #
    #             # Check settings
    #             print('Mode: ', mode)
    #             print('Augmentation Mode: ', aug_mode)
    #             print('File Path: ', raw_data)
    #             print('Sampling Rate: ', sr)
    #             exit()
    #
    #             # Generate folders
    #             folder_name = raw_data.split('\\')[3]
    #             print('Folder Name: ', folder_name)
    #
    #             #