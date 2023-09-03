#predicting the results from racing at haydock
#data is from taken from https://github.com/joenano/rpscrape

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
import seaborn as sns

#pd read csv hayock_edited.csv and use the first row as the column names
data=pd.read_csv('haydock_edited.csv', header=0)


# make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

#look at the terms used in the data
#for col in data.columns:
#    print("these are the variables:   ", col)


#fill 0's in the na data
data=data.fillna(0)

#convert object into integer data ['or'] and ['rpr']
data['or']=pd.to_numeric(data['or'], errors='coerce')
data['rpr']=pd.to_numeric(data['rpr'], errors='coerce')



#one hot encoding for the features ['class'], ['age_band'],  ['going'], ['horse'] , ['sex'] , ['jockey'] , ['trainer']
#using sklearn
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
data['class'] = onehotencoder.fit_transform(data[['class']]).toarray()
data['age_band'] = onehotencoder.fit_transform(data[['age_band']]).toarray()
data['going'] = onehotencoder.fit_transform(data[['going']]).toarray()
data['horse'] = onehotencoder.fit_transform(data[['horse']]).toarray()
data['sex']= onehotencoder.fit_transform(data[['sex']]).toarray()
data['jockey']= onehotencoder.fit_transform(data[['jockey']]).toarray()
data['trainer']= onehotencoder.fit_transform(data[['trainer']]).toarray()

#print("this is after one hot enconding: " , data['going'])



#the pos column is the label so we need to remove it from the data
features=data.drop(['pos'], axis=1)

print("these are the features: ")
print(features.head())



#the label

#convert all string in ['pos'] column into a integer
data['pos']=pd.to_numeric(data['pos'], errors='coerce')
#convert all na values into 0
data['pos']=data['pos'].fillna(0)
data['pos']=data['pos'].astype(int)


#the pos column is the label
label=data['pos']
#print("this is the label head", label.head())

#putting the data into numpy arrays

label_array=data['pos'].values
print("this is the label array")
print(label_array)



#only want the winners for the y value so anything over 1.1 is a 0 and anything under is a 1
label_array=np.where(label_array>1.1,0,1)

#print the features and label data types
print("this is the features data type")
print(features.dtypes)
print("this is the label data type")
print(label_array.dtype)





#split the data into train and test
train_features, test_features = train_test_split(features, test_size=0.2)

#split the label into train and test
train_label, test_label= train_test_split(label_array, test_size=0.2)


print ("this is the train: ", train_features.head())
print ("this is the test: ", test_features.head())

n_features=train_features.shape[1]

print (n_features)
print("this is the number of features", n_features)

#create a sns pairplot of the features
sns.pairplot(train_features[['class','age_band','dist_m','going','ran','lbs','jockey','trainer','or','rpr']],diag_kind="kde")
plt.show()



#creating the model

# define model
model = Sequential()
model.add(Dense(150, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(75, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(9, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(train_features, train_label, epochs=5)

# evaluate the model
loss, acc = model.evaluate(test_features, test_label)
print('Test Accuracy: %.3f' % acc)


# make a prediction

#twothirty=([[2,7,1,0 ,0 ,8,165,170,181 ],
#[2,7,2,0,0,7,165,161,172],
#[2,7,3,0,0,8,165,149,165],
#[2,7,4,0,0,8,165,165,174],
#[2,7,5,0,0,8,165,149,159],
#[2,7,6,0,0,9,157,147,167],
#[2,7,7,0,0,8,157,153,168]])

#what = model.predict([twothirty])
#print("this is the prediction:  ", what[0])






