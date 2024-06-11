#taken from AI and Machine Learning for Coders
import tensorflow as tf

#load the data into train and test data
data=tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

#print the dimensions of training_images
print("dimensions before reshaping and normalising", training_images.ndim)

#reshape the images_for_classification to 28x28
training_images=training_images.reshape(60000, 28, 28, 1)
testing_images=test_images.reshape(10000, 28, 28, 1)

#normalize the images_for_classification
training_images=training_images / 255.0
testing_images=testing_images / 255.0

#print the dimensions of training_images after reshaping
print("dimensions after reshaping and normalising", training_images.ndim)

#build the model, uses 2 convolutional layers and 2 pooling layers then flattens the data and uses 2 dense layers 128 and 10

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

#compile the model with adam optimizer and sparse categorical crossentropy loss function and accuracy metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#fit the model with 10 epochs with validation data
history = model.fit(training_images, training_labels, epochs=50, validation_data=(testing_images, test_labels))

#evaluate the model
model.evaluate(testing_images, test_labels)


# list all data in history
print(history.history.keys())

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predict the model with test_images
classifications=model.predict(testing_images)

#print all the predictions
print(classifications[0])

#print the label of the predictions
print(test_labels[0])

