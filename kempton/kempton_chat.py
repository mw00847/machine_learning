import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('kempton_edited.csv')

# Perform one-hot encoding for categorical columns
categorical_columns = ['class', 'age_band', 'going','sex', 'jockey', 'trainer']
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoder.get_feature_names(categorical_columns)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

#convert any non numeric values in 'or' to 0
data['or'] = pd.to_numeric(data['or'], errors='coerce')
data['or'] = data['or'].fillna(0)

#convert any non numeric values in 'rpr' to 0
data['rpr'] = pd.to_numeric(data['rpr'], errors='coerce')
data['rpr'] = data['rpr'].fillna(0)



# Drop the original categorical columns and concatenate the encoded ones
data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)

# Convert 'pos' and 'ran' columns to numeric
data['pos'] = pd.to_numeric(data['pos'], errors='coerce')
data['ran'] = pd.to_numeric(data['ran'], errors='coerce')

# Remove rows where 'pos' is over 12 and 'ran' is under 12
data = data[(data['pos'] <= 12) & (data['ran'] >= 12)]

# Define the number of rows per sequence (12 in this case)
sequence_length = 12

# Create numpy arrays for features and target
features = []
target = []

for i in range(0, len(data) - sequence_length + 1, sequence_length):
    sequence = data.iloc[i:i + sequence_length]  # Get a sequence of 12 rows
    features.append(sequence.drop('pos', axis=1).values)
    target.append(sequence['pos'].values)

features = np.array(features)
target = np.array(target)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=50)  # Adjust the number of components as needed
features_pca = pca.fit_transform(features.reshape(-1, features.shape[-1]))

# Reshape features to have a consistent 3D shape
features_pca = features_pca.reshape(-1, sequence_length, pca.n_components)

#plot the results from the PCA
import matplotlib.pyplot as plt
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


#plot PC1 and PC2
plt.scatter(features_pca[:,0], features_pca[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#plot PC1 and PC3
plt.scatter(features_pca[:,0], features_pca[:,2])
plt.xlabel('PC1')
plt.ylabel('PC3')
plt.show()

#plot PC2 and PC3
plt.scatter(features_pca[:,1], features_pca[:,2])
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.show()

#plot PC1 and PC4
plt.scatter(features_pca[:,0], features_pca[:,3])
plt.xlabel('PC1')
plt.ylabel('PC4')
plt.show()

#plot PC2 and PC4
plt.scatter(features_pca[:,1], features_pca[:,3])
plt.xlabel('PC2')
plt.ylabel('PC4')
plt.show()

#plot PC3 and PC4
plt.scatter(features_pca[:,2], features_pca[:,3])
plt.xlabel('PC3')
plt.ylabel('PC4')

