#https://www.kaggle.com/datasets/davidregan/greyhound-racing-uk-predict-finish-position


import pandas as pd
import numpy as np

#set pandas to print all the data
pd.set_option('display.max_columns', None)

#read the data from the csv file
data=pd.read_csv('crayford_results.csv', header=0)

print(data.head())

#normalize each of the columns
data = (data - data.mean()) / (data.max() - data.min())

#make all the columns apart from finished the features
features=data.drop(['Finished'], axis=1)

target = ['Finished']

# Features
features = ['Trap', 'BSP', 'Time_380', 'Finish_Recent', 'Finish_All', 'Stay_380',\
            'Races_All','Odds_Recent','Odds_380', 'Distance_Places_All', 'Dist_By',\
            'Races_380', 'Last_Run','Early_Time_380', 'Early_Recent' ,\
            'Distance_All', 'Wins_380', 'Grade_380','Finish_380','Early_380',\
            'Distance_Recent', 'Wide_380', 'Favourite']

#organise the features and the target data
all_labels = data[target].values
x_features = data[features].values


#print the length of all_labels
print("this is the length of all_labels", len(all_labels))

#create a numpy array for each 6 traps of each race
y = []
for i in range(0, len(all_labels), 6):
    y.append([all_labels[i:i+6]])
y = np.array(y).reshape(2001, 6)
x = []
for i in range(0, len(x_features), 6):
    x.append([x_features[i:i+6:]])
x = np.array(x).reshape(2001, 6, -1)

#print the shape of y
print("!this is the shape of y", np.shape(y))






#split the data into training and testing data

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

#create the model

model = Sequential()
model.add(BatchNormalization(input_shape=x_train.shape[1:]))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(6, activation='softmax'))
model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['accuracy'])
print(model.summary())


#fit the model
history = model.fit(x_train, y_train, epochs=100)




import matplotlib.pyplot as plt

# Assuming you have 'history' containing accuracy and loss data

# Convert accuracy to percentage
accuracy_percentage = [acc * 100 for acc in history.history['accuracy']]

# Convert loss to percentage
loss_percentage = [loss * 100 for loss in history.history['loss']]

# Plot training accuracy
plt.plot(accuracy_percentage, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training Accuracy')

plt.show()

# Plot training loss
plt.plot(loss_percentage, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (%)')
plt.legend()
plt.title('Training Loss')

plt.show()

