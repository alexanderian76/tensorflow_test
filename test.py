import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

loaded_model = tf.keras.models.load_model('test_model.keras')

tokenizer = Tokenizer(num_words=10000)
s = tokenizer.texts_to_sequences(["Watch this, it is not waste of time"])
p = pad_sequences(s, maxlen=50, padding='post')

print(loaded_model.predict(p))