#predict the surface tension of a molecule using descriptors


import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors

#molecules of interest
#water, MPG, BG, glycerine, benzylOH, MEG, DEG, methanol

# Define the SMILES representations of the compounds
smiles = ["O", "CC(CO)O", "CCCCOCCO", "C(C(CO)O)O", "C1=CC=C(C=C1)CO", "C(CO)O", "C(COCCO)O", "CO"]

#surface tension values
SFT=[72.8, 45.6, 37.6, 63.1, 39.2, 48.9, 44.2, 22.6]

#print the length of the SFT list
print(len(SFT))

# Create a list to store the calculated descriptors for each compound
descriptors_list = []

# Calculate descriptors for each compound
for smile in smiles:
    molecule = Chem.MolFromSmiles(smile)
    descriptors = [descriptor[1](molecule) for descriptor in Descriptors.descList]
    descriptors_list.append(descriptors)

# Convert the descriptors to a numpy array
descriptors_array = np.array(descriptors_list)

#print("these are the descriptors",descriptors_array)

#descriptors array is the features and the SFT list is the target

#target property is the surface tension
target_properties = np.array([SFT]) # Example target properties
print(target_properties)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Assuming you have descriptors_array and target_properties as defined before
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    descriptors_array, target_properties, test_size=0.2, random_state=42
)

# Create a TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),  # Adjust num_features based on your actual data
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression, 1 neuron for the predicted value
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)

print("Mean Squared Error on test data:", loss)

# Now you can use the trained model to make predictions on new data
#predictions = model.predict(X_test)