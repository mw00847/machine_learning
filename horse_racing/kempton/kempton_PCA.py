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

#looking at the data
train=pd.read_csv('kempton_train.csv', names=column_names)
train=train.fillna(0)
train=train.astype(int)
#if pos is not 1 then it is 0
train['pos']=np.where(train['pos']!=1,0,1)

print(train.head())

#take away all but the or and pos columns
train=train[['pos','ovr_btn']]

#normalise the data before PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[['ovr_btn']] = scaler.fit_transform(train[['ovr_btn']])

#run pca on the data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf.head())

#plot the data
plt.figure(figsize=(10,10))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title("Principal Component Analysis of Kempton Dataset")
targets = [0, 1]
colors = ['r', 'g']
#loop through the targets and colors
for target, color in zip(targets,colors):
    indicesToKeep = train['pos'] == target
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()