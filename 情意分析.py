import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
from tensorflow.keras.datasets.imdb import get_word_index

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)
model = Sequential()
model.add(Embedding(10000, 128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
word_index = get_word_index()
print(word_index['this'])
text = "this movie is bad"
seq = [word_index[x] for x in text.split()]
print(model.predict([seq]))
model_json = model.to_json()
open('imdb_model_architecture.json', 'w').write(model_json)
model.save_weights('ismdb_model_weights.h5')
