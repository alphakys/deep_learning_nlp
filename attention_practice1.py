import warnings

import keras

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from keras.datasets import imdb
from keras.utils import to_categorical
from keras.utils import pad_sequences

vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

len_list = [len(l) for l in X_train]

max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

import tensorflow as tf
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from keras import Input, Model
from keras import optimizers
import os

class BahdanauAttention(keras.Model):
    def __int__(self, units):
        super(BahdanauAttention, self).__int__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    # 키와 value는 같음
    def call(self, values, query):
        # query shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해 차원을 변경해줌
        hidden_with_time_axis = tf.expand_dims(query, 1)







