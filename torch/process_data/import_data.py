import os 
import re
import argparse
import pandas as pd 
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_spectrogram(args):
    mode = args.mode 
    if mode == "Train":
        if os.path.exists('Train_Spectogram_Images'):
            return
        # Get Genres and Track IDs from the tracks.csv file
        filename_metadata = "fma_metadata/tracks.csv"
        tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
        tracks_array = tracks.values
        tracks_id_array = tracks_array[: , 0]
        tracks_genre_array = tracks_array[: , 40]
        tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
        tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)

        folder_sample = "fma_small"
        directories = [d for d in os.listdir(folder_sample)
                       if os.path.isdir(os.path.join(folder_sample, d))]
        counter = 0
        print("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Train_Spectogram_Images'):
            os.makedirs('Train_Spectogram_Images')
        for d in directories:
            label_directory = os.path.join(folder_sample, d)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith(".mp3")]

            # Convert .mp3 files into mel-Spectograms
            for f in file_names:
                track_id = int(re.search(r'fma/.*/(.+?).mp3', f).group(1))
                track_index = list(tracks_id_array).index(track_id)
                if(str(tracks_genre_array[track_index, 0]) != '0'):
                    print(f)
                    y, sr = librosa.load(f)
                    melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
                    mel = librosa.power_to_db(melspectrogram_array)
                    # Length and Width of Spectogram
                    fig_size = plt.rcParams["figure.figsize"]
                    fig_size[0] = float(mel.shape[1] / 100)
                    fig_size[1] = float(mel.shape[0] / 100)
                    plt.rcParams["figure.figsize"] = fig_size
                    plt.axis('off')
                    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
                    librosa.display.specshow(mel,cmap='gray_r')
                    plt.savefig("Train_Spectogram_Images/"+str(counter)+"_"+str(tracks_genre_array[track_index,0])+".jpg", dpi=100)
                    plt.close()
                    counter = counter + 1
        return

    elif mode == "Test":
        if os.path.exists('Music_Spectogram_Images'):
            return
        folder_sample = "../templates/music/"
        counter = 0
        print("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Music_Sepctogram_Images'):
            os.makedirs('Music_Spectogram_Images')
        file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample)
                       if f.endswith(".mp3")]
        # Convert .mp3 files into mel-Spectograms
        for f in file_names:
            test_id = re.search(r'../templates/music/(.+?).mp3', f).group(1)
            print(f)
            y, sr = librosa.load(f)
            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)
            # Length and Width of Spectogram
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel,cmap='gray_r')
            plt.savefig("Music_Spectogram_Images/"+test_id+".jpg", dpi=100)
            plt.close()
        return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Converts the files mp3 into mel-spectrograms')
    argparser.add_argument('-m', '--mode', required=True, help='set mode to process data for Train or Test')
    args = argparser.parse_args()
    create_spectrogram(args)
