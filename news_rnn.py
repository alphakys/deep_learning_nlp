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
from keras.layers import Dense, Embedding, LSTM

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
