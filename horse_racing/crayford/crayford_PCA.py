import pandas as pd
import numpy as np

#set pandas to print all the data
pd.set_option('display.max_columns', None)

#read the data from the csv file
data=pd.read_csv('crayford_results.csv', header=0)

#normalize each of the columns
data = (data - data.mean()) / (data.max() - data.min())

#perform PCA on the data
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)
print(pca.n_features_)
print(pca.n_components_)
print(pca.n_samples_)
print(pca.mean_)

#plot the pca results
import matplotlib.pyplot as plt
plt.plot(pca.explained_variance_ratio_)
plt.show()

#plot PC1 and PC2
plt.scatter(pca.components_[0], pca.components_[1])
plt.show()

#plot PC1 and PC3
plt.scatter(pca.components_[0], pca.components_[2])
plt.show()

#plot PC1 and PC4
plt.scatter(pca.components_[0], pca.components_[3])
plt.show()
