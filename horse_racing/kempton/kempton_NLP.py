import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the combined text file and make the first the column header
data = pd.read_csv('kempton_NLP_data.csv')



# Preprocess the data

#fill 0's in the na data
data=data.fillna(0)

# For the data.pos column if the value is not 1 then it is 0
data['pos'] = np.where(data['pos'] != 1, 0, data['pos'])

# Convert data.comment to lower case
data['comment'] = data['comment'].str.lower()

# Remove noise
import re

# Remove special characters
data['comment'] = data['comment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

# Remove punctuation
import string
data['comment'] = data['comment'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Stemming/lemmatization
# Example using NLTK for stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def stem_text(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# Note: You would typically stem before tokenizing in a real workflow
data['comment'] = data['comment'].apply(stem_text)

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.4, random_state=42)

# Tokenize the data for both training and validation sets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['comment'])

train_sequences = tokenizer.texts_to_sequences(train_data['comment'])
val_sequences = tokenizer.texts_to_sequences(val_data['comment'])

max_sequence_length = max(len(seq) for seq in train_sequences)

# Pad sequences for both training and validation sets
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)
val_padded_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length)

# Convert labels to the appropriate data type
train_labels = train_data['pos'].astype('float32')
val_labels = val_data['pos'].astype('float32')

# Define and compile the model
model = Sequential([
    Flatten(input_shape=(max_sequence_length,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_padded_sequences, train_labels, batch_size=32, epochs=10, validation_data=(val_padded_sequences, val_labels))

# Evaluate the model
loss, accuracy = model.evaluate(val_padded_sequences, val_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

