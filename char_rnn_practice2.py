import os
import warnings
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import tensorflow as tf

from string import punctuation

from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

from keras.utils import pad_sequences, to_categorical
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, TextVectorization
from keras.models import Sequential

raw_text = '''
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
'''

tokens = raw_text.split()
sentences = ' '.join(tokens)

index_to_char = {idx: v for idx, v in enumerate(sorted(list(set(sentences))))}
char_to_index = {v: idx for idx, v in enumerate(sorted(list(set(sentences))))}

char_size = len(char_to_index) + 1

cnt = len(sentences)
sep_length = 10
preprocessing_X = []
for _ in range(cnt):
    sep_arr = sentences[_:sep_length + _]
    preprocessing_X.append([char_to_index[char] for char in sep_arr])

train_X = pad_sequences(preprocessing_X)[:, :-1]
y = pad_sequences(preprocessing_X)[:, -1]

y = to_categorical(y, num_classes=char_size)

embedding_output_dim = 40
hidden_units = 256

model = Sequential()
model.add(Embedding(char_size, embedding_output_dim))
model.add(LSTM(hidden_units))
model.add(Dense(units=char_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(train_X, y, epochs=80)


def sentence_generation(epochs=10):
    input_dim = 9
    start = np.random.randint(428)

    test_X = np.zeros(shape=(input_dim + epochs, input_dim))
    test_sen = sentences[start:start+input_dim]
    test_char = [char_to_index[char] for char in test_sen]

    prd_sentence = test_sen
    for i in range(epochs):

        test_sentence = np.expand_dims(test_char, axis=0)
        prd = np.argmax(model(test_sentence))
        prd_sentence += index_to_char[prd]
        np.append(test_sentence[0], prd)[1::]

    return prd_sentence