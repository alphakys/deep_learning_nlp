import os
import warnings

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from keras.datasets import reuters
from keras.layers import SimpleRNN, Dense, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, pad_sequences

# num_words param means only include top ? most frequent words in dataset
# test_split param means split ratio of train and test dataset
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

num_classes = max(y_train) + 1

train_length = [len(l) for l in X_train]

avr_length = int(sum(train_length) / len(train_length))
max_length = max(train_length)

fig, axe = plt.subplots(ncols=1)
# bins means number of bar it measn how many groups you want to make
plt.hist(train_length, bins=50)
plt.xlabel('legnth of samples')
plt.ylabel('number of samples')
plt.show()

fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(12, 5)
sns.countplot(y_train)

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
# reuters dataset is ordered by frequency of words
# but index should be added 3 it is because the first 3 index is reserved for special tokens
index_to_word = {v+3: k for k,v in reuters.get_word_index().items()}

index_to_word[0] = "<pad>"
index_to_word[1] = "<sos>"
index_to_word[2] = "<unk>"

# total train data word length is 1000
vocab_size = 1000
# max length of train data word is 100.
# this will be used for padding
max_len = 100

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

embedding_dim = 120
hidden_units = 128
# total label number in true data set
num_classes = 46


# [STUDY] index_to_word에서 숫자 0은 패딩을 의미하는 토큰인 pad, 숫자 1은 문장의 시작을 의미하는 sos, 숫자 2는 OOV를 위한 토큰인 unk라는 특별 토큰에 맵핑되어져야 합니다
#   reuters만의 rule이 있다.!!!
model = Sequential()

model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(hidden_units))
model.add(Dense(num_classes, activation='softmax'))
# callback function
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=64, epochs=30, callbacks=[es], validation_data=(X_test, y_test))
