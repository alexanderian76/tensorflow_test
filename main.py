import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load your dataset from a CSV file
file_path = 'imdb_labelled.csv'  # Replace with your file path
data = pd.read_csv(file_path, header=None, sep='\t', names=["Comment", "isPositive"])

texts = data.pop("Comment")
labels = data
# Preprocess the text data
tokenizer = Tokenizer(num_words=10000)  # Limit vocabulary to the top 10,000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # Convert text to sequences of integers

# Pad sequences to ensure uniform input size
max_length = 50  # Maximum length of each sequence
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to numpy array
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),  # Embedding layer
    layers.GlobalAveragePooling1D(),  # Pooling layer to reduce dimensionality
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0)

s = tokenizer.texts_to_sequences(["The worst movie I've been watching ever! Actors are fcing silly"])
p = pad_sequences(s, maxlen=max_length, padding='post')

# Evaluate the model on the test data
print(model.predict(p))