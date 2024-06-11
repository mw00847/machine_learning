#predicting horse racing results
#data scraped using https://github.com/joenano/rpscrape

#the data has already been split into test and train

#Importing libraries
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

#sorting out the data, filling 0 for na

train=pd.read_csv('kempton_train.csv', names=column_names)
train=train.fillna(0)
train=train.astype(int)

test=pd.read_csv('kempton_test.csv', names=column_names)
test=test.fillna(0)
test=test.astype(int)

#print(train.head())
#print(test.head())

#the features are the first 9 columns
#putting the data into numpy arrays
train_x=train.iloc[:,0:9].values
test_x=test.iloc[:,0:9].values

#the target labels are the pos the last column
train_y=train.iloc[:,9].values
test_y=test.iloc[:,9].values

#print the train_y and test_y before the change
#print("this is the test data", test_y)
#print("this is the train data", train_y)

#only want the winners for the y value so anything over 1.1 is a 0 and anything under is a 1
test_y=np.where(test_y>1.1,0,1)
train_y=np.where(train_y>1.1,0,1)

train_x=np.array(train_x)
test_x=np.array(test_x)

#print(test_y)
#print(train_y)

#print the number of features in train
#n_features=train_x.shape[1]

#print n_features
#print("this is the number of features", n_features)


#normalise the features, x
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
train_x=scaler.fit_transform(train_x)
test_x=scaler.fit_transform(test_x)

print("this is the normalised train_x", train_x)
print("this is the normalised test_x", test_x)

#run PCA on the features
from sklearn.decomposition import PCA
pca=PCA(n_components=9)
train_x=pca.fit_transform(train_x)
print("this is the pca train_x", train_x)

#plot the pca data
#plt.scatter(train_x[:,0], train_x[:,1], c=train_y, cmap='winter')
#plt.title("PCA 1 vs PCA 2")
#plt.show()

#plt.scatter(train_x[:,1], train_x[:,2], c=train_y, cmap='hot')
#plt.title("PCA 2 vs PCA 3")
#plt.show()


#build the model

model=Sequential()
model.add(Dense(9, input_dim=9, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
history=model.fit(train_x, train_y, epochs=10, batch_size=32)
#could use validation data here?


#evaluate the model
loss, acc=model.evaluate(test_x, test_y, verbose=0)
print("this is the accuracy", acc)

#plot the loss and accuracy
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train')
#plt.plot(history.history['val_accuracy'], label='test')

plt.legend()
plt.show()

