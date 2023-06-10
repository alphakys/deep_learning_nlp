import os
import warnings

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
from keras.callbacks import EarlyStopping, ModelCheckpoint

news_pd: pd.DataFrame = pd.read_csv('ArticlesJan2018.csv')
not_unknow_indices = news_pd['headline'] != 'Unknown'
headline_list = news_pd[not_unknow_indices]
headline_list = headline_list['headline'].tolist()


# punctutation 제거와 loswer case로 변환
def get_preprocessed_data(raw_sentences):
    processed_sentences = raw_sentences.encode('utf8').decode('ascii', 'ignore')
    return processed_sentences


preprocessed_headline = [get_preprocessed_data(headline) for headline in headline_list]

tk = Tokenizer()
tk.fit_on_texts(headline_list)

vocab_size = len(tk.word_index) + 1

sequences = []
for line in headline_list:
    sequence = tk.texts_to_sequences([line])[0]
    sequences.append(sequence)

encoded_list = []
for seq in sequences:
    cnt = len(seq)
    for i in range(2, cnt):
        encoded_list.append(seq[:i])

max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(encoded_list)

X = padded_sequences[:, :-1].astype('float32')
y = padded_sequences[:, -1]
# !!! y one hot encoding을 수행한다.
y = to_categorical(y, num_classes=vocab_size)


def train_model(X, y):
    embedding_output_dim = 10
    hidden_units = 128

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_output_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0017), loss='categorical_crossentropy', metrics=['accuracy'])

    # [STUDY] min_delta 개선된 것으로 간주할 최소한의 변화량
    #   min_delta==0.001이라면 0.001보다 작은 변화량은 개선이 없는 것으로 간주한다. 0.001도 변화한 량으로 간주하여 training을 이어간다.
    #   patience: 개선이 없는 에포크를 얼마나 기다릴 것인가
    #   baseline: 모델이 달성해야 할 최소한의 기준값을 설정한다.
    # cb_es = [EarlyStopping(monitor='loss', baseline=0.95, patience=5, min_delta=0.01)]
    # cb_mc = ModelCheckpoint(monitor='accuracy', save_best_only=True
    history = model.fit(X, y, epochs=200)

    return model, history


def predict_word(model, tk, search: str, n: int):
    # tokenizer에서 lower로 변환하기 때문에 마찬가지로 lower 함수를 써서 convert한다.
    test_index = tk.word_index[search.lower()]
    # 검색된 리스트 중에서 첫번째를 테스트 sentence로 한다.

    # true_X = sequences[:3]
    search_list = X[:, -1]

    valid_index = int([idx for idx, w in enumerate(search_list) if w == test_index][0])

    valid_inputs = X[valid_index:valid_index + n]

    true_word = ''
    prd = model.predict(valid_inputs)
    for w in prd:
        index = np.argmax(w)
        true_word += tk.index_word[index] + ' '

    return true_word.rstrip()

# 어제 복습
# 따라서 핵심은 input_dim = vocab_size

# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Embedding(output_dim=vocab_size, input_length=)

# [STUDY] inputs에 들어가는 value가 Embedding layer의 input_dim보다 작아야 한다가 핵심이다.
#   np.random.randint(1000, size=(32, 10))에서 의미하는 바는 input의 정수 최댓값은 1000보다 작다는 것을 의미하고
#   embedding layer가 하는 역할은 결국 to_categorical처럼 encoding을 해주는 것인데 만약 들어온 input값이 설정한 vocab_size보다 크면
#   encoding을 할 수가 없다.
#   또한 integer가 1000보다 작아야 한다는 것이다.

# inputs = np.random.randint(10, size=(4, 5))
# [STUDY] 중요!!!! 한마디로 embedding의 첫번째 파라미터는 vocab_size을 설정하면 된다.
#   그리고 두번째 파라미터는 output_dim을 설정하면 된다.(Dense의 units와 같은 개념이다.)
# e = Embedding(input_dim=10, output_dim=7)
