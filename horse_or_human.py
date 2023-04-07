#Import Libraries
import tensorflow as tf
import tensorflow_datasets as tfds #Dataset API
import numpy as np #Linear Algebra
import matplotlib.pyplot as plt #Data visualization
import os #Manipulate Files
from PIL import Image #Manipulate Images


from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=(1./255), #Rescales pixel values (originally 0-256) to 0-1
    )

print ("this is train_datagen", train_datagen)



train_generator = train_datagen.flow_from_directory(
    'Data/Train',
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.Sequential([

tf.keras. layers. Conv2D(16, (3,3), activation='relu',
input_shape=(300, 300, 3)),
tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Conv2D(32, (3,3), activation='relu'), tf.keras. layers.MaxPooling2D (2,2), tf.keras.layers.Conv2D(64, (3,3), activation='relu'), tf.keras. layers.MaxPooling2D(2,2), tf.keras.layers.Conv2D(64, (3,3), activation='relu'), tf.keras. layers.MaxPooling2D(2,2), tf.keras.layers.Conv2D(64, (3,3), activation='relu'), tf. keras.layers.MaxPooling2D(2,2), tf .keras. layers.Flatten(),
tf.keras. layers. Dense (512, activation='relu'), tf.keras. layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',
	optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
	metrics=['accuracy'])

#Plots model training history

history=model.fit_generator(
	train_generator,
	epochs=15
)

plt.show()

