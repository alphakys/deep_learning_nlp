import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import pandas as pd
from pandas import DataFrame

from keras.datasets import mnist
from decimal import Decimal
from keras.layers import SimpleRNN, Bidirectional, LSTM, TextVectorization, Dense, Dropout, Activation, Input
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.optimizers import Adam

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from nltk.tokenize import word_tokenize

# [STUDY] to_categorical 함수는 정수형 클래스 레이블을 one hot encoding 벡터로 변환하는 함수입니다.
# to_categorical 함수는 정수형 클래스 레이블을 one hot encoding 벡터로 변환하는 함수입니다.
# to_categorical 함수는 정수형 클래스 레이블을 one hot encoding 벡터로 변환하는 함수입니다.


text = """경마장에 있는 말이 뛰고 있다\n그의 말이 법이다\n가는 말이 고와야 오는 말이 곱다"""

tk = Tokenizer()
prepared_text = [text]

tk.fit_on_texts(prepared_text)
splited_text = text.split('\n')

vocab_size = list(tk.word_index.values())[-1] + 1

text_sequences = [tk.texts_to_sequences([sen]) for sen in splited_text]
index_to_word = {v:k for k,v in tk.word_index.items()}

train_X = [to_categorical(seq, vocab_size)[:, :, 1::] for seq in text_sequences]

time_steps = []
y = []
j = 1
while j < 5:
    tmp = text_sequences[0][0][:j]
    y.append(text_sequences[0][0][j])
    time_steps.append(tmp)
    j += 1

train_y = to_categorical(y, vocab_size)

padded_seq = pad_sequences(time_steps)
padded_seq = pad_sequences(time_steps, padding='post')
expanded_timesteps = np.expand_dims(padded_seq, axis=0).reshape(4, 4, -1)

oh_time_steps = to_categorical(expanded_timesteps, vocab_size)
oh_time_steps[:, :, 0].fill(0)

true_word = [index_to_word[index] for index in y]
test_words = [index_to_word[t] for t in text_sequences[0][0]]

inputs = oh_time_steps
model = Sequential()
model.add(SimpleRNN(100, input_shape=inputs.shape[1::], return_sequences=True))
model.add(SimpleRNN(200, activation='tanh'))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(inputs, train_y, epochs=50, batch_size=2)

def get_true(index):
    print(f"테스트 단어 : \n{test_words[index]}")
    true_idx = np.argmax(model.predict(np.expand_dims(inputs[index], axis=0), verbose=False))
    print(f"예측 단어 : \n{index_to_word[true_idx]}")



# input_tensor = Input(shape=inputs.shape)

# test_text = text.split('\n')[0]
#
# tk = Tokenizer()
# tk.fit_on_texts([test_text])
#
# from nltk.tokenize import word_tokenize
# tokens = word_tokenize(test_text)
#
# word_matrix = tk.texts_to_matrix(tokens)
# word_matrix = word_matrix[:,1::]
# train_X = word_matrix[:3]
# train_X = np.array(train_X)
# # train_X = np.array(train_X).reshape(-1, train_X.shape[0], train_X.shape[1])
#
# y = word_matrix[3]
#
# model = Sequential()
#   [STUDY] DEEP NEURAL NETWORK를 하고자 하면 return_sequences=True를 해줘야지 다음층으로 은닉층의 출력값이 넘어가게 된다.
# model.add(SimpleRNN(5, input_shape=train_X.shape, return_sequences=True))
# model.add(SimpleRNN(5, activation='tanh'))
# model.add(Dense(5, activation='softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_X, y, epochs=100, batch_size=1)
# model.evaluate(train_X, y)


# cancer_data = load_breast_cancer()
#
# cancer_df = pd.DataFrame(data=cancer_data['data'], columns=cancer_data['feature_names'])
# y = cancer_data['target']
#
# ss = StandardScaler()
# processed_dataset = ss.fit_transform(cancer_df)
#
# X_train, X_test, y_train, y_test = train_test_split(processed_dataset, y, random_state=0)
#
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
#
# prd = lr.predict(X_test)
# score = accuracy_score(y_test, prd)
#
# cm = confusion_matrix(y_true=y_test, y_pred=prd)

#
# ads_pd = pd.read_csv('Social_Network_Ads.csv')
# ads_pd['Gender'] = ads_pd['Gender'].transform(lambda x: float(1) if x == 'Male' else float(0))
#
# X = ads_pd.iloc[:, 1:-1]
# y = ads_pd.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# ss = StandardScaler()
# # processed_X = ss.fit_transform(X_train)
#
# lr = LogisticRegression()

# x_train = x_train.values.reshape(-1, 1)
# x_test = x_test.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# y_test = y_test.values.reshape(-1, 1)
#

#
# processed_x_train = ss.fit_transform(x_train)
# processed_x_test = ss.fit_transform(x_test)
#
# lr.fit(processed_x_train, y_train)
# lr.fit()


# sentence = 'What wii the fat cat sit on'
#
# from keras.preprocessing.text import text_to_word_sequence
# import tensorflow as tf
#
# token = text_to_word_sequence(sentence)
#
# import numpy as np
#
# np.linspace(0, 1, 10)
# a = np.linspace(0, 1, 10)
# a = np.linspace(0.1, 0.9, 50)
#
# from matplotlib import pyplot as plt
#
# a = np.linspace(0.001, 0.97, 50)
# a = np.linspace(0.001, 0.97, 500)
#
# a = np.linspace(0.001, 0.9999, 500)
# y = [v / (1 - v) for v in a]
# y = np.array(y)
# y = np.log(y)
# plt.plot(a, y, 'r')
# plt.plot(y, a, 'r')
# plt.plot(y, a, 'b')


# 로짓함수와 로지스틱 함수의 관계 및 로지스틱 회귀분석에 대한 전반적인 이해
# 한 50퍼센트 이해한거 같다. 조금만 더 공부하면 될거 같다.
# 이제 한 80퍼센트 이해한거 같다.!!


# [STUDY] padding을 할 때, 몇 개의 padding을 할 것인가?를 위해 구하는 최종 행렬
#   x = target_shape[0] -1 + filter_shape[0]


# [STUDY]
#  C order란 ROW MAJOR ORDER를 의미한다.
#  F order란 COLUMN MAJOR ORDER를 의미한다.
# n_arr = np.ndarray(shape=(10, 10), dtype=int, buffer=np.array(range(10)), order='C', offset=1)

# [STUDY]
#   결국에는 vectorization이란 cpu가 수행할 명령어 세트를 instructon이라고 하면 이 instruction을 수행할 데이터가
#   하나 by 하나라면 시간이 오래 걸릴 것인데
#   modern cpu는 하나의 instruction을 처리하는데 여러 multiple data를 가지고 와서도 처리할 수 있게 되었다는 점
#   register에 담을 수 있는 operand 그릇이 여러개니까 여러 데이터를 가져와서 한 명령어에서 수평적으로 처리할 수 있게 한다
#   그리고 numpy에서는 결국 이와 같은 연산을 수행하기 위해서 만약에 길이가 다른 두 array가 있다면 vectorization을 시키기 위해서
#   작은 array의 길이를 copy하여 길이를 맞춰 주고 그릇에 여러 데이터를 한 번에 담아서 operation을 수행한다는 점.
#    ** VECTORIZATION **
#     The software changes required to exploit instruction level parallelism are know ans vectorization
#    Single Instruction Multiple Data
#    ** pipe line **
#   유닉스 계열 운영 체제에서(어느 정도까지는 마이크로소프트 윈도우에서) ㅔㅈ공되는 병행성 매커니즘의 하나로서, 두 프로세스가 생산자-소비자 모델에 따라 통신할 수 있게 해주는 원형
#   버퍼이다. 즉 파이프는 한 프로세스가 쓰고 다른 프로세스가 읽는 선입선출 형태의 큐라 할 수 있다. 파이프의 개념은 코루틴으로부터 영향을 받아 만들어졌으며, ㅎ운영 체제 기술의 발전에 큰 공헌을
#   print(하였다)
#    파이프에는 일정한 크기의 공간이 할당되어 있다. 어떠 ㄴ프로세스가 파이프에 데이터를 기록하려고 할 때, 충분한 공간이 남아있다면 기록이 즉시 수행되겠지만, 공간이 부족하다면 그 프로세스는
#    print(차단된다)이것은 웅여체제가 상호배제를 수행한 결과이다.
