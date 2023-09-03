#predicting properties of molecules with rdkit


import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors

# Define the SMILES representation of benzene
benzene_smiles = "c1ccccc1"

# Generate the RDKit molecule object from SMILES
molecule = Chem.MolFromSmiles(benzene_smiles)

# Calculate descriptors for the molecule
descriptors = []
for descriptor_name, descriptor_function in Descriptors.descList:
    descriptor_value = descriptor_function(molecule)
    descriptors.append(descriptor_value)

# Convert the descriptors to a numpy array
descriptors = np.array(descriptors)

# Generate a random target property for training
target_property = np.random.rand()

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(len(descriptors),)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with benzene descriptors and target property
model.fit(descriptors.reshape(1, -1), np.array([target_property]), epochs=10)