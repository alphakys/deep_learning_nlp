import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('spam.csv', encoding='latin1')
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

data['v1'] = data['v1'].replace(['spam', 'ham'], [1, 0])
data.info()

data.drop_duplicates(subset=['v2'], inplace=True)

from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

X_data = data['v2']
y_data = data['v1']

# [STUDY] stratify 중요!!!
#   stratify를 적용해야지만 label의 분포가 고르게 분포된다는 점!!
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

tk = Tokenizer()
tk.fit_on_texts(X_train)

threshhold = 2
# 등장 개수가 threshhold(임계점) 보다 작은 단어의 count
rare_cnt = 0
total_cnt = len(tk.word_index)
# 훈련 데이터의 전체 단어 빈도수 총합
total_freq = 0
# 훈련 데이터의 희귀 단어 빈도수 총합
rare_freq = 0

for item in tk.word_counts.items():
    if item[1] < 2:
        rare_cnt += 1

tk = Tokenizer(num_words=total_cnt - rare_cnt + 1)
tk.fit_on_texts(X_train)

X_train_encoded = tk.texts_to_sequences(X_train)
X_test_encoded = tk.texts_to_sequences(X_test)

max_len = max([len(seq) for seq in X_train_encoded])
vocab_size = len(tk.word_index) + 1

from keras.utils import pad_sequences

X_train = pad_sequences(X_train_encoded, maxlen=max_len)
X_test = pad_sequences(X_test_encoded, maxlen=max_len)

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Embedding, Dense, SimpleRNN

embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(SimpleRNN(units=hidden_units))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

