import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.datasets import imdb
from keras.utils import pad_sequences

vocab_size = 20000

# imdb 데이터 로드 -> num_words는 빈도수가 높은 상위 몇 번째 데이터까지를 로드할 것인지를 설정하는 parameters
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# train data(각 리뷰)의 word 길이를 저장하는 리스트
train_list_len = np.array(list(map(lambda x: len(x), X_train)))
# test data(각 리뷰)의 word 길이를 저장하는 리스트
test_list_len = np.array(list(map(lambda x: len(x), X_test)))

train_list_length_average = np.mean(train_list_len, dtype=int)
test_list_length_average = np.mean(test_list_len, dtype=int)

max_len = 400

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))

# 모든 단어 벡터의 평균을 구한다.
# [STUDY] 중요! GlobalAveragePooling에서
#   input_shape = TensorShape([25000, 400, 64])이라면
#   250000, 64로 column(아래로) axis로 평균이 구해진다.
#   즉 [[2,3,4],
#       [3,4,6]] 이면 GlobalAveragePooling시 [2.5, 3.5, 5] 이렇게 column에 대해서 평균 값을 구해서 반환 받는다.
#   결국 globalAveragePooling의 의미는 각 embedding vector의 차원이 10차원이라고 예를 든다면
#   각 차원이 단어에서 의미하는 바가 정확하게 모르지만 각 단어의 각 차원은 유사한 기능을 할 것이라는 가정이 들어가 있지 않을까 추정된다(나의 생각)

model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('embedding_average_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.2)