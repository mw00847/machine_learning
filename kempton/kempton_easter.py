
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.optimizers import Adam


# Configure GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#pd read csv kempton_edited.csv and use the first row as the column names
data=pd.read_csv('kempton_edited.csv', header=0)

#sorting the data

#fill 0's in the na data
data=data.fillna(0)
#are there na values in data
print("are there na values in data: ", data.isna().values.any())

#set pandas to print all columns
pd.set_option('display.max_columns', None)

#convert all values in data ['rpr'] and ['or'] that are not numbers or could be - to 0
data['rpr']=pd.to_numeric(data['rpr'], errors='coerce')
data['or']=pd.to_numeric(data['or'], errors='coerce')


print("this is the data head after organising the data", data.head())



#check from here

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
data['owner']= onehotencoder.fit_transform(data[['owner']]).toarray()

print("this is after one hot enconding: " , data['going'])
print("after one hot encoding the data looks like this: ")
print(data.head())



print("this is the data describe", data.describe())

#the pos column is the label so we need to remove it from the data
features=data.drop(['pos'], axis=1)

print("these are the features: ")
print(features.head())


#the labels

#convert all string in ['pos'] column into a integer
data['pos']=pd.to_numeric(data['pos'], errors='coerce')
#convert all na values into 0
data['pos']=data['pos'].fillna(0)
data['pos']=data['pos'].astype(int)

#the pos column is the label
label=data['pos']
#print("this is the label head", label.head())

label_array=data['pos'].values
#only want the winners for the y value so anything over 1.1 is a 0 and anything under is a 1

label_array=np.where(label_array>1.1,0,1)
print("this is the label array")
print(label_array)


#change -values to 0 in the rpr column of the data array
data['rpr']=np.where(data['rpr']<0,0,data['rpr'])

#

#describe the data
print("this is the data description: ")
print(data.describe())



#describe the number of features
n_features=features.shape[1]

features=features.values


#split the data into train test and validation

from sklearn.model_selection import train_test_split

# Split the data into train and test sets for features
train_features, test_features, train_label_array, test_label_array = train_test_split(features, label_array, test_size=0.4, random_state=42)

# Further split the train set into train and validation sets
train_features, valid_features, train_label_array, valid_label_array = train_test_split(train_features, train_label_array, test_size=0.5, random_state=42)




#creating the model

# define model
model = Sequential()
model.add(Dense(10000, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(5000, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(2500, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1250, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))


# Define optimizer with a custom learning rate
custom_learning_rate = 0.001 # Change this value as desired
optimizer = Adam(learning_rate=custom_learning_rate)

# Compile the model with the custom optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ... (previous code for data preprocessing and model building)

# Train the model
history = model.fit(train_features, train_label_array, epochs=10, validation_data=(valid_features, valid_label_array))


# Plot training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot training accuracy and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

