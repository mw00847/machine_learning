import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#bring in the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#name the categorical and numerical features
cat_features=['Pclass','Sex','Embarked']
num_features=['Age','SibSp','Parch','Fare']

#remove the n/a values from the data using SimpleImputer from sklearn
from sklearn.impute import SimpleImputer

def preprocess (df,num_features,cat_features,dv):
    features=num_features+cat_features
    #selecting the target variable
    if dv in df.columns:
        y=df[dv]
    else:
        y=None

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df[cat_features]=imp_mode.fit_transform(df[cat_features] )
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[num_features]=imp_mean.fit_transform(df[num_features])

    X=pd.get_dummies(df[features],columns=cat_features,drop_first=True)
    return X,y



#run the function to use simpleimputer for both test and train
y, X =  preprocess(train, num_features, cat_features, 'Survived')
test_y, test_X = preprocess(test, num_features, cat_features, 'Survived')




# PCA on the dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=4)
principal_components = pca.fit_transform(X)

principal_dataframe = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4'])

finalDf = pd.concat([principal_dataframe, y], axis=1)

# Plot the principal components
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)


targets = y
colors = ['r', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Survived'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
               finalDf.loc[indicesToKeep, 'PC2'],
               c=color, s=50)

ax.legend(targets)
ax.grid()
plt.show()