import tensorflow as tf
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv('cheltenham.csv')
#print(data.head())

#drop the rows with missing values

data=data.dropna()
#print(data.head())

print(data.dtypes)

#name columns after the data.dytpes
#columns=['going','ran','num','pos','ovr_btn','btn','age','sex','lbs','time','secs','dec','or','rpr','jockey','trainer']

#assign the columns to the data
print ("these are the columns", data.columns)

#one hot encoding

print("before encoding", data['going'].unique())



one_hot_encoder = OneHotEncoder(sparse=False)

#one hot encoding with array.reshape(-1, 1)
one_hot_encoder.fit(data['going'].array.reshape(1, -1))

print("after encoding", one_hot_encoder.fit(data['going'].array.reshape(1, -1))

