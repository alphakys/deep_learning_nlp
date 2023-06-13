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
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, TextVectorization, TimeDistributed
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

f = open('11-0.txt', 'rb')
sentences_list = []

# text preprocessing
for sentence in f:
    sentence = sentence.strip()  # strip을 통해 \r, \n을 제거한다.
    sentence = sentence.lower()  # 소문자화
    sentence = sentence.decode('ascii', 'ignore')  # ascii 코드로 변환
    if len(sentence) > 0:
        sentences_list.append(sentence)
f.close()

# [STUDY] 하나 배움 .join()!!!!
raw_sentences = ' '.join(sentences_list)

char_vocab = sorted(list(set(raw_sentences)))
vocab_size = len(char_vocab)

char_to_index = dict((char, index) for index, char in enumerate(char_vocab))

index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key

model = load_model('char_rnn.h5', compile=False)

def sentence_generation(model, length):
    # 문자에 대한 랜덤한 정수 생성
    ix = [np.random.randint(vocab_size)]

    # 랜덤한 정수로부터 맵핑되는 문자 생성
    y_char = [index_to_char[ix[-1]]]
    print(ix[-1],'번 문자',y_char[-1],'로 예측을 시작!')

    # (1, length, 55) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성
    X = np.zeros((1, length, vocab_size))

    for i in range(length):
        # X[0][i][예측한 문자의 인덱스] = 1, 즉, 예측 문자를 다음 입력 시퀀스에 추가
        X[0][i][ix[-1]] = 1
        print(index_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(index_to_char[ix[-1]])
    return ('').join(y_char)



seq_length = 60

# 문자열의 길이를 seq_length로 나누면 전처리 후 생겨날 샘플 수
n_samples = int(np.floor((len(raw_sentences) - 1) / seq_length))

train_X = []
train_y = []

for i in range(n_samples):
    # 0:60 -> 60:120 -> 120:180로 loop를 돌면서 문장 샘플을 1개씩 pick.
    X_sample = raw_sentences[i * seq_length: (i + 1) * seq_length]

    # 정수 인코딩
    X_encoded = [char_to_index[c] for c in X_sample]
    train_X.append(X_encoded)

    # 오른쪽으로 1칸 쉬프트
    y_sample = raw_sentences[i * seq_length + 1: (i + 1) * seq_length + 1]
    y_encoded = [char_to_index[c] for c in y_sample]
    train_y.append(y_encoded)

train_X = to_categorical(train_X)
train_y = to_categorical(train_y)

hidden_units = 256

model = Sequential()
model.add(LSTM(hidden_units, input_shape=(None, train_X.shape[2]), return_sequences=True))
model.add(LSTM(hidden_units, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=80, verbose=1)


#
# vocab_size = len(tk.word_index) + 1
#
# sequences = []
# for text in sentences_list:
#     split_text = text.split(' ')
#     sequence = [tk.texts_to_sequences(text) for text in split_text]
#     for seq in sequence:
#         tmp_seq = np.array(seq).flatten()
#         sequences.append(tmp_seq)
#
# max_len = max([len(seq) for seq in sequences])
# padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
#
# X = padded_sequences[:, :-1]
# y = padded_sequences[:, -1]
# y = to_categorical(y, num_classes=vocab_size)
#
# # [STUDY] Embedding layer는 input_dim, output_dim이 중요하다.
# #   output_dim은 Dense layer의 unit과 같다. -> 하나의 단어에 대해서 몇개의 multiverse를 가질 것인가의 개념
# #   input_dim은 inputs data의 총 단어수의 + 1 즉 vocab_size와 같아야한다.
# #   embedding layer는 Dense()에 activation이 linear인 layer와 같다. API이기 때문에 쉽게 사용할 수 있도록 만들어진 함수에 불과함
# #   결국! 주어진 input 데이터에 weights를 곱하여 embedding layer를 만드는데 가장 핵심적인 부분은
# #   input_dimentioality이다. 왜냐하면 input_shape=(10, 5) 이면 weights는 (5, output_dim)이기 때문에
# #   input_dim이 5로 동일하든지 그 보다 작아야만 한다.
#
# embedding_dim = 10
# hidden_units = 128
#
# model = Sequential()
# # embedding에 input_length == input_dim과 같다.
# # !!! [STUDY] embedding layer에서 내가 vocab_size 만큼 input_dim을 늘려주지 않아도 알아서 처리한다.
# model.add(Embedding(vocab_size, embedding_dim))
# model.add(LSTM(hidden_units))
# model.add(Dense(units=vocab_size, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
# history = model.fit(X, y, epochs=10, batch_size=16)
#
#
# def predict_word(model, tk, n: int):
#     pred_sentence = ''
#     rand = np.random.randint(1, vocab_size)
#     rand_word = tk.index_word[rand]
#
#     print(f"random word => {rand_word}")
#     curr_word = rand_word
#     for i in range(n):
#         test_input = pad_sequences(tk.texts_to_sequences([curr_word]), maxlen=max_len - 1, padding='pre')
#         prd = model(test_input)
#         pred_idx = np.argmax(prd)
#         pred_char = tk.index_word[pred_idx]
#
#         pred_sentence += pred_char
#         curr_word = pred_char
#     return pred_sentence