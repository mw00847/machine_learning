import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#initialize the gpu
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Set pandas to print all the data
pd.set_option('display.max_columns', None)

# Read the data from the csv file
data = pd.read_csv('crayford_results.csv')




# Normalize each of the columns
data = (data - data.mean()) / (data.max() - data.min())

# Define the features and target
features = ['Trap', 'BSP', 'Time_380', 'Finish_Recent', 'Finish_All', 'Stay_380',
            'Races_All', 'Odds_Recent', 'Odds_380', 'Distance_Places_All', 'Dist_By',
            'Races_380', 'Last_Run', 'Early_Time_380', 'Early_Recent', 'Distance_All',
            'Wins_380', 'Grade_380', 'Finish_380', 'Early_380', 'Distance_Recent',
            'Wide_380', 'Favourite']
target = ['Finished']

#print the features head 
print(data[features].head())


# Organize the features and the target data
all_labels = data[target].values
x_features = data[features].values

# Create a numpy array for each 6 traps of each race
y = []
for i in range(0, len(all_labels), 6):
    y.append([all_labels[i:i+6]])
y = np.array(y).reshape(2001, 6)
x = []
for i in range(0, len(x_features), 6):
    x.append([x_features[i:i+6:]])
x = np.array(x).reshape(2001, 6, -1)

# Split the data into training, validation, and testing data
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.40)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.50)

model = Sequential()
model.add(BatchNormalization(input_shape=x_train.shape[1:]))
model.add(Dense(16, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(6, activation='softmax'))
model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(x_train, y_train, epochs=300, validation_data=(x_val, y_val))

# Plot the training accuracy and loss
accuracy_percentage = [acc * 100 for acc in history.history['accuracy']]
loss_percentage = [loss * 100 for loss in history.history['loss']]
plt.plot(accuracy_percentage, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training Accuracy')
plt.show()
plt.plot(loss_percentage, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (%)')
plt.legend()
plt.title('Training Loss')
plt.show()

#save the plots
plt.savefig('crayford2_accuracy.png')
plt.savefig('crayford2_loss.png')
