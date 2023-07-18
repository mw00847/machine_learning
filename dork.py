import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pathlib
import matplotlib.pyplot as plt


#use pathlib to get the path to the data to images for tensorflow
data_dir = pathlib.Path('faces')


print(len(os.listdir('faces/Rose')))

rose= list(data_dir.glob('Rose/*'))

grace = list(data_dir.glob('Grace/*'))

mat = list(data_dir.glob('Mat/*'))

wendy = list(data_dir.glob('Wendy/*'))

floyd = list(data_dir.glob('Floyd/*'))


#class names are the folder names
class_names = os.listdir('faces')
print("these are the class names: ", class_names)

print("the length of the data is ", len(mat))

image_count = len(list(data_dir.glob('*/*.jpg')))
print("total number of images = : ", image_count)


train_ds = tf.keras.utils.image_dataset_from_directory(
  'faces',
  validation_split=0.8,
  subset="training",
  seed=123,
  image_size=(224,224),
  batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'faces',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(224,224),
  batch_size=32)

print("before rescaling", val_ds)

#normalise the tensorflow data set to 0-1
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./224)

print(normalization_layer)

print("after rescaling", val_ds)