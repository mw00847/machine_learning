#import modules

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
data = pd.read_csv('lingfield_data.csv')

#print data head and type of data in each column
print(data.head())
print(data.dtypes)

#print all the columns that are objects
print(data.select_dtypes(include=['object']).columns)

objects = data.select_dtypes(include=['object']).copy()
#print(objects.head())

#drop some columns
objects.drop(['comment'], axis=1, inplace=True)

#one hot encode the objects
objects = pd.get_dummies(objects, columns=['horse', 'jockey', 'trainer','owner'])

#print all the columns in objects
print(objects.columns)


