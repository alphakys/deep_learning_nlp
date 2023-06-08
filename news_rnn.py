import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd

from string import punctuation

from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.layers import Dense, Embedding, LSTM, SimpleRNN
from keras.models import Sequential


news_pd: pd.DataFrame = pd.read_csv('ArticlesJan2018.csv')

headline_list = news_pd['headline'].tolist()

tk = Tokenizer()
tk.fit_on_texts(headline_list)

vocab_size = len(tk.word_index) + 1

sequences = []
for line in headline_list:
    sequence = tk.texts_to_sequences([line])[0]
    sequences.append(sequence)

encoded_list = []
for seq in sequences[:3]:
    cnt = len(seq)
    for i in range(2, cnt):
        encoded_list.append(seq[:i])

max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(encoded_list)

X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

model = Sequential()

embedding_size = 10



model.add()

def predict_word(model, tk, search: str, n: int):
    
    # tokenizer에서 lower로 변환하기 때문에 마찬가지로 lower 함수를 써서 convert한다.
    test_index = tk.word_index[search.lower()]
    # 검색된 리스트 중에서 첫번째를 테스트 sentence로 한다.
    index_arr = [s for s in sequences if test_index in s][0]

    # maxx = len(index_arr)

    # print(index)
















