import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import urllib.request
from matplotlib import pyplot as plt
from collections import Counter
from konlpy.tag import Okt, Mecab
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename='steam.txt')

# read_table can read tab separated data and names -> coumns name
total_data = pd.read_table('steam.txt', names=['label', 'reviews'])

total_data.drop_duplicates(subset=['reviews'], inplace=True)  # reviews 열에서 중복인 내용이 있다면 중복 제거

train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
train_data['label'].value_counts().plot(kind='bar')

# 한글과 공백을 제외하고 모두 제거
train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['reviews'].replace('', np.nan, inplace=True)

test_data.drop_duplicates(subset=['reviews'], inplace=True)  # 중복 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지',
             '임', '게', '만', '게임', '겜', '되', '음', '면']

mecab = Mecab()

train_data['tokenized'] = train_data['reviews'].apply(lambda x: mecab.morphs(x))
train_data['tokenized'] = [token for token in train_data['tokenized'] if token not in stopwords]
test_data['tokenized'] = test_data['reviews'].apply(lambda x: mecab.morphs(x))
test_data['tokenized'] = [token for token in test_data['tokenized'] if token not in stopwords]

negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)

X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test= test_data['tokenized'].values
y_test = test_data['label'].values

tk = Tokenizer()
tk.fit_on_texts(X_train)

threshold = 2
total_cnt = len(tk.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tk.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

vocab_size = total_cnt - rare_cnt + 2
max_len = 60

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

import re
from keras.layers import Embedding, Dense, LSTM, Bidirectional
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
embedding_dim = 100
hidden_units = 128
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units)))

model.add(Dense(1, activation='sigmoid'))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es], batch_size=128, validation_split=0.2)








