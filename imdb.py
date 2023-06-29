import os
import warnings
import numpy as np

from matplotlib import pyplot as plt
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.callbacks import EarlyStopping

warnings.simplefilter(action='ignore', category=FutureWarning)

vocab_size = 10000
max_len = 500
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GRU(hidden_units))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
es = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')

history = model.fit(X_train, y_train, batch_size=32, epochs=30, callbacks=[es], validation_data=[X_test, y_test])