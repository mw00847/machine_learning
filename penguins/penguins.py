

#penguin classification problem

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#load the data
starting_dataset, info = tfds.load('penguins', split='train', with_info=True)
#print(info)

#the penguin names are the class names
penguin_names = ['Adelie', 'Chinstrap', 'Gentoo']

#download the preprocessed data
ds_split,info=tfds.load('penguins',split=['train[:80%]','train[80%:]'],with_info=True,as_supervised=True)


#split the data into training and testing
ds_test=ds_split[0]
ds_train=ds_split[1]

#assert that the data is in the correct format
assert isinstance(ds_train, tf.data.Dataset)


#print(ds_train)
#print(ds_test)

df_train = tfds.as_dataframe(ds_train, info)
df_test = tfds.as_dataframe(ds_test, info)

#print(df_train.head())
#print(df_test.head())

#make the batch size 32
ds_train_batch  = ds_train.batch(32)

#iter through the train_batch
features, labels = next(iter(ds_train_batch))
print(features)
print(labels)

#plot some clusters
plt.scatter(features[:,0],
            features[:,3],
            c=labels,
            cmap='winter')
plt.show()


#build the model with input shape of 4
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
])




