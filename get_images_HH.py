#human or horse

#https://www.kaggle.com/code/calebreigada/tensorflow-image-classification-guide



#Import Libraries
import tensorflow as tf
import tensorflow_datasets as tfds #Dataset API
import numpy as np #Linear Algebra
import matplotlib.pyplot as plt #Data visualization
import os #Manipulate Files
from PIL import Image #Manipulate Images

import warnings
warnings.filterwarnings('ignore') #ignores warnings

#Make sure Tensorflow is version 2.0 or higher
print('Tensorflow Version:', tf.__version__)

# Makes Folders to store images
os.makedirs('Data', exist_ok=True)
os.makedirs('Data/Train/Horses', exist_ok=True)
os.makedirs('Data/Train/Humans', exist_ok=True)
os.makedirs('Data/Test/Horses', exist_ok=True)
os.makedirs('Data/Test/Humans', exist_ok=True)

base_path = os.getcwd()
horse_counter = 0
human_counter = 0
# The below code will save the dataset images into the folders created above
# Note: This step is not required when using Tensorflow datasets but will be required when
# using datasets that are in the wild or possibly on Kaggle
# see horse or humans doc here ->https://www.tensorflow.org/datasets/catalog/horses_or_humans
for i, dataset in enumerate(tfds.load('horses_or_humans', split=['train', 'test'])):
    if i == 0:  # training set
        set_path = os.path.join(base_path, 'Data/Train')
    else:  # test set
        set_path = os.path.join(base_path, 'Data/Test')

    for row in list(dataset):
        im = Image.fromarray(row['image'].numpy())
        if row['label'] == 0:  # 0 is horse and 1 is human
            class_path = os.path.join(set_path, 'Horses')
            file_path = os.path.join(class_path, "horse_{}.jpeg".format(horse_counter))
            horse_counter += 1
        elif row['label'] == 1:  # 0 is horse and 1 is human
            class_path = os.path.join(set_path, 'Humans')
            file_path = os.path.join(class_path, "human_{}.jpeg".format(horse_counter))
            human_counter += 1
        im.save(file_path)  # saves the image in the proper folder



print('Number of Horse Images in the Training Set:', len(os.listdir('Data/Train/Horses')))
print('Number of Human Images in the Training Set:', len(os.listdir('Data/Train/Humans')))
print('\n')
print('Number of Horse Images in the Testing Set:', len(os.listdir('Data/Test/Horses')))
print('Number of Human Images in the Testing Set:', len(os.listdir('Data/Test/Humans')))

