import os
import re
import argparse
from PIL import Image

"""
Slice the spectrogram into multiple 128x128 images which will be the input to the
Convolutional Neural Network.
"""
def slice_spect(args):
    mode = args.mode 
    if mode=="Train":
        if os.path.exists('Train_Sliced_Images'):
            return
        labels = []
        image_folder = "Train_Spectogram_Images"
        filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                       if f.endswith(".jpg")]
        counter = 0
        print("Slicing Spectograms ...")
        if not os.path.exists('Train_Sliced_Images'):
            os.makedirs('Train_Sliced_Images')
        for f in filenames:
            genre_variable = re.search(r'Train_Spectogram_Images/.*_(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            width, height = img.size
            number_of_samples = width / subsample_size
            for i in range(round(number_of_samples)):
                start = i*subsample_size
                img_temporary = img.crop((start, 0., start + subsample_size, 128))
                img_temporary.save("Train_Sliced_Images/"+str(counter)+"_"+genre_variable+".jpg")
                counter += 1
        return

    elif mode=="Test":
        if os.path.exists('Music_Sliced_Images'):
            return
        labels = []
        image_folder = "Music_Spectogram_Images"
        filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                       if f.endswith(".jpg")]
        counter = 0
        print("Slicing Spectograms ...")
        if not os.path.exists('Music_Sliced_Images'):
            os.makedirs('Music_Sliced_Images')
        for f in filenames:
            song_variable = re.search(r'Music_Spectogram_Images/(.+?).jpg', f).group(1)
            img = Image.open(f)
            subsample_size = 128
            width, height = img.size
            # Take 20 frames from the 6th frame to calculate the cosine similarity
            number_of_samples = 26
            for i in range(6, number_of_samples):
                start = i*subsample_size
                img_temporary = img.crop((start, 0., start + subsample_size, subsample_size))
                img_temporary.save("Music_Sliced_Images/"+str(counter)+"_"+song_variable+".jpg")
                counter = counter + 1
        return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Slices the spectrograms into 128x128 slices')
    argparser.add_argument('-m', '--mode', required=True, help='set mode to process data for Train or Test')
    args = argparser.parse_args()
    slice_spect(args)