import tensorflow as tf

#the data comes from keras datasets
data=tf.keras.datasets.fashion_mnist
(training_images,training_labels), (test_images, test_labels) = data.load_data()

print(training_images.shape)
print(training_labels.shape)


#the data is 28*28 numpy array with pixel values ranging from 0 to 255
#normalize the pixel values
training_images=training_images/255.0
test_images=test_images/255.0

#build the model
#flatten the images_for_classification into a 1D array
model = tf.keras.models.Sequential([
tf.keras. layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras. layers.Dense(10, activation=tf.nn.softmax)
])


#compile the model with a learning rate of 0.01 using the Adam optimizer
#and sparse_categorical_crossentropy as the loss function

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
loss='sparse_categorical_crossentropy',metrics=['accuracy'])



#train the model
model.fit(training_images, training_labels, epochs=10)

#print out how many images_for_classification in the test_images array



#make predictions using the test images_for_classification
classifications=model.predict(test_images)
print(classifications[0])
print(test_labels[0])

