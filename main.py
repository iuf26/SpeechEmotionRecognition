# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import librosa
import librosa.display as lr
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from io import BytesIO
import requests
import os
import tensorflow as tf
import pathlib
import datetime

# creates a model.json and model.h5 file containing the CNN details
def alexNetModel(datasetPath):
    data_dir = pathlib.Path(datasetPath)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    # classnames in the dataset specified
    CLASS_NAMES = ['anger' ,'anxiety' ,'boredom' ,'disgust' ,'happiness' ,'neutral' ,'sadness']
    print(CLASS_NAMES)
    # print length of class names
    output_class_units = len(CLASS_NAMES)
    print(output_class_units)
    model = tf.keras.models.Sequential([
        # 1st conv
        tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
        # 2nd conv
        tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        # 3rd conv
        tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        # 4th conv
        tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        # 5th Conv
        tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
        # To Flatten layer
        tf.keras.layers.Flatten(),
        # To FC layer 1
        tf.keras.layers.Dense(4096, activation='relu'),
        # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
        # To FC layer 2
        tf.keras.layers.Dense(4096, activation='relu'),
        # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_class_units, activation='softmax')
    ])
    model_json = model.to_json()
    with open("AlexnetModel/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("AlexnetModel/model.h5")
    print("Saved model to disk")

def remasterEmoDbDataset(rawDataSetPath, remasteredDatasetPath):
    emotionInitialInGermanMappedToEmotionFolderName = {
        "F": "happiness",
        "W": "anger",
        "L": "boredom",
        "A": "anxiety",
        "T": "sadness",
        "E": "disgust",
        "N": "neutral"
    }
    directory = os.fsencode(rawDataSetPath)
    idx = 0

    for file in os.listdir(directory):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        filename = os.fsdecode(file)
        fullWavFilePath = rawDataSetPath + '\\' + filename
        y, sr = librosa.load(fullWavFilePath, duration=3)
        mfcc = librosa.feature.melspectrogram(y=y)
        mfcc_log_mel = librosa.power_to_db(mfcc, ref=np.max)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        librosa.display.specshow(mfcc_log_mel)
        plt.savefig("img1.jpg", bbox_inches='tight')
        librosa.display.specshow(mfcc_delta)
        plt.savefig("img2.jpg",bbox_inches='tight')
        librosa.display.specshow(mfcc_delta2)
        plt.savefig("img3.jpg",bbox_inches='tight')
        plt.close()
        red = Image.open("img1.jpg")
        green = Image.open("img2.jpg")
        blue = Image.open("img3.jpg")
        combined = Image.merge('RGB', (red.getchannel('R'), green.getchannel('G'), blue.getchannel('B')))
        filename = filename.split('.')[0]
        emotionLetter = ''.join(c for c in filename if c.isupper())
        combined = combined.resize((227,227))
        combined.save(remasteredDatasetPath + '\\' + emotionInitialInGermanMappedToEmotionFolderName[emotionLetter] + '\\' + filename + ".jpg")
        idx = idx + 1
        print(idx)
    os.remove('img1.jpg')
    os.remove('img2.jpg')
    os.remove('img3.jpg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rawDataSetPath = 'D:\\LICENTA\\Database\\EMODB\\wav'
    remasteredDatasetPath = 'D:\\LICENTA\\Database\\EMODB-GROUPED' #create a folder for each emotion label in EMODB
    remasterEmoDbDataset(rawDataSetPath, remasteredDatasetPath)
    #alexNetModel(remasteredDatasetPath)   call this function to create de AlexnetModel locally

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
