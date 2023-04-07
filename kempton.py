#predicting the results from racing on boxing day at kempton

# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense



# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

#the column names are the variables
column_names=['dist', 'ran', 'num', 'ovr_btn', 'btn', 'age', 'lbs', 'or', 'rpr','pos']

#sorting out the data

train=pd.read_csv('kempton_train.csv',names=column_names) 
train=train.fillna(0) 
train=train.astype(int)     

test=pd.read_csv('kempton_test.csv',names=column_names) 
test=test.fillna(0)
test=test.astype(int)


#putting the data into numpy arrays

train_x=train.iloc[:,0:9].values
test_x=test.iloc[:,0:9].values


train_y=train.iloc[:,9].values
test_y=test.iloc[:,9].values

test_y=np.where(test_y>1.1,0,1) 
train_y=np.where(train_y>1.1,0,1) 

train_x=np.array(train_x)
test_x=np.array(test_x)

#print(train_x)
#print(train_y)


n_features=train_x.shape[1]


# define model
model = Sequential()
model.add(Dense(150, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(75, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(9, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(train_x, train_y, epochs=5)

# evaluate the model
loss, acc = model.evaluate(test_x, test_y)
print('Test Accuracy: %.3f' % acc)


# make a prediction

twothirty=([[2,7,1,0 ,0 ,8,165,170,181 ],
[2,7,2,0,0,7,165,161,172],
[2,7,3,0,0,8,165,149,165],
[2,7,4,0,0,8,165,165,174],
[2,7,5,0,0,8,165,149,159],
[2,7,6,0,0,9,157,147,167],
[2,7,7,0,0,8,157,153,168]])

what = model.predict([twothirty])
print("this is the prediction:  ", what[0])














