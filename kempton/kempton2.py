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

train=pd.read_csv('kempton_train.csv', names=column_names)
train=train.fillna(0)
train=train.astype(int)

test=pd.read_csv('kempton_test.csv', names=column_names)
test=test.fillna(0)
test=test.astype(int)



print("data in the pandas dataframe", train.head())
print(test.head())


#putting the data into numpy arrays

train_x=train.iloc[:,0:9].values
test_x=test.iloc[:,0:9].values


train_y=train.iloc[:,9].values
test_y=test.iloc[:,9].values

#print the train_y and test_y before the change
print("this is the test data", test_y)
print("this is the train data", train_y)

#only want the winners for the y value so anything over 1.1 is a 0 and anything under is a 1

test_y=np.where(test_y>1.1,0,1)
train_y=np.where(train_y>1.1,0,1)

train_x=np.array(train_x)
test_x=np.array(test_x)

#print test y and train y
print("this is the y data, then into numpy array", test_y)
print("this is the y data", train_y)

#print train_x  and test_x
print("this is the x data", test_x)
print("this is the x data", train_x)


#print(train_x)
#print(train_y)


n_features=train_x.shape[1]

#print n_features
print("this is the number of features", n_features)
