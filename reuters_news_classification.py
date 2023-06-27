import os
import warnings

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from keras.datasets import reuters
from keras.layers import SimpleRNN, Dense, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# num_words param means only include top ? most frequent words in dataset
# test_split param means split ratio of train and test dataset
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

max_len = max([len(v) for v in X_train])
from keras.utils import pad_sequences

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# [STUDY] index_to_word에서 숫자 0은 패딩을 의미하는 토큰인 pad, 숫자 1은 문장의 시작을 의미하는 sos, 숫자 2는 OOV를 위한 토큰인 unk라는 특별 토큰에 맵핑되어져야 합니다
#   reuters만의 rule이 있다.!!!
model = Sequential()
embedding_dim = 50
hidden_units = 32

model.add(Embedding(30982, embedding_dim, input_length=max_len))
model.add(SimpleRNN(hidden_units))
num_classes = 46

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = model.fit(X_train, y_train, batch_size=64, epochs=30, callbacks=[es], validation_data=(X_test, y_test))
