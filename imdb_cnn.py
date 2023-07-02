import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from keras.datasets import imdb
from keras.utils import pad_sequences

vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

embedding_dim = 256  # 임베딩 벡터의 크기
hidden_units = 128  # 은닉층 units 개수 -> 뉴런
dropout_ratio = 0.3  # 과적합을 방지하기 위한 dropout 비율 => 일부 은닉층은(0.3의 비율) 활설화 되지 않는다.
kernel_size = 3  # filter의 사이즈 -> feature map을 그리기 위한 element wise 연산에 사용할 필터의 크기
num_filters = 256  # filter의 개수

from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Dropout(dropout_ratio))

# An activation function is the last component of the convolutional layer to increase the non-linearity in the output.
# [STUDY] CONVOLUTIONAL ACTIVATION => https://towardsdatascience.com/beginners-guide-to-understanding-convolutional-neural-networks-ae9ed58bb17d
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es], epochs=20)








