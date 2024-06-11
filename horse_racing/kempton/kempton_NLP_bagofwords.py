import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('kempton_NLP_data.csv')

#fill 0's in the na data
data=data.fillna(0)

import string
import re

# stemming/lemmatization
# Example using NLTK for stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def stem_text(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# Preprocess the data
data['comment'] = data['comment'].str.lower()
data['comment'] = data['comment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
data['comment'] = data['comment'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
data['comment'] = data['comment'].apply(stem_text)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['pos'], test_size=0.2, random_state=42)

# Vectorize the text data using Bag of Words
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Define and train a logistic regression model
model = LogisticRegression(max_iter=1000)
#model=LogisticRegression(solver=saga)
model.fit(X_train_bow, y_train)

# Evaluate the model
y_pred = model.predict(X_test_bow)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
