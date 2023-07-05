import time
import timeit
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import nltk
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 토큰화에 품사 태깅이 된 데이터 받아오기
tagged_sentences = nltk.corpus.treebank.tagged_sents()

sentences, pos_tags = [], []
for tagged_sentence in tagged_sentences:  # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence)  # 각 샘플에서 단어들은 sentence에 품사 태깅 정보들은 tag_info에 저장한다.
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(list(tag_info))  # 각 샘플에서 품사 태깅 정보만 저장한다.

max_len = max([len(l) for l in sentences])
tk = Tokenizer()
tk.fit_on_texts(sentences)

vocab_size = len(tk.word_index) + 1
sequences = tk.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_len)

pos_tk = Tokenizer()
pos_tk.fit_on_texts(pos_tags)

pos_sequences = pos_tk.texts_to_sequences(pos_tags)
pos_pad_seq = pad_sequences(pos_sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, pos_pad_seq, test_size=0.25, random_state=1004)

from keras.models import Model
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate, LSTM, Bidirectional


inputs = Input(shape=(X_train.shape[1]))
embedding_dim = 128
hidden_units = 128
num_filters = 512
dropout_ratio = 0.3
num_classes = len(pos_tk.word_index)

x = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
x = Bidirectional(LSTM(hidden_units, return_sequences=True))(x)

conv_list = []
kernel_size = [3, 5, 7]

for sz in kernel_size:
    conv = Conv1D(kernel_size=sz,
                  filters=num_filters,
                  padding='valid',
                  activation='relu',
                  strides=1)(x)
    conv = GlobalMaxPooling1D()(conv)
    print(conv.shape)
    conv_list.append(conv)

output = Concatenate()(conv_list) if len(conv_list) > 1 else conv_list[0]
output = Dropout(dropout_ratio)(output)
model_output = Dense(num_classes, activation='softmax')(output)

model = Model(inputs, model_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_split=0.2)
