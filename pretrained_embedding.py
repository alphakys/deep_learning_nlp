import os
import pathlib
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten



from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import gzip
import zipfile

sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality',
             'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]

tk = Tokenizer()
tk.fit_on_texts(sentences)
vocab_size = len(tk.word_index) + 1

X_encoded = tk.texts_to_sequences(sentences)
max_len = max([len(l) for l in X_encoded])
X_train = pad_sequences(X_encoded)
y_train = np.array(y_train)

embedding_dim = 8

import gensim

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/alpha/문서/GoogleNews-vectors-negative300.bin', binary=True)

word2vec_model.get_vector()


# embedding_dict = {}
# f = open('/home/alpha/문서/glove.6B/glove.6B.100d.txt')
#
# for line in f:
#
#     word_vector = line.split()
#     word = word_vector[0]
#
#     word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
#     embedding_dict[word] = word_vector_arr
#
# f.close()
#
# embedding_matrix = np.zeros((vocab_size, 100))
#
# for token, idx in tk.word_index.items():
#     vector_val = embedding_dict.get(token, None)
#     if vector_val is not None:
#         embedding_matrix[idx] = vector_val
#
# # embedding layer를 사용할 때의 embedding_dim과 같음 단지 pretrained embedding이기 때문에 다른 용어를 사용하는 듯
# output_dim = 100
#
# model = Sequential()
# # pretrained embedding(glove)을 사용하기 때문에 trainable=False로 설정한다.
# # 거듭 생각하자. input_length -> time_steps이다.
# # X_train.shape => (7,4) 그리고 7이 batch 사이즈였다.
# e = Embedding(vocab_size, output_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)
# model.add(e)
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(X_train, y_train, epochs=100, verbose=1)


# [STUDY] Embeddingd (number of samples, input_length) 이때 각 sample은 정수 인코딩이 된 결과로 정수 시퀀스입니다.
#   output으로 (batch_size, input_length, output_dim)의 3D 텐서를 리턴합니다.
# sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality',
#              'bad', 'highly respectable']
# y_train = [1, 0, 0, 1, 1, 0, 1]
#
# tk = Tokenizer()
# tk.fit_on_texts(sentences)
# vocab_size = len(tk.word_index) + 1
#
# embedding_dim = 8
#
# X_encoded = tk.texts_to_sequences(sentences)
# max_len = max([len(l) for l in X_encoded])
# X_train = pad_sequences(X_encoded)
# y_train = np.array(y_train)
#
# model = Sequential()
# # [STUDY] input_length -> time_steps --> 한 문장에서의 token의 길이
# #   현재 학습하는 dataset에서는 각각의 리스트가 하나의 샘플로 구성된다.(batch)
# # Embedding에서 vocab_size를 알고자 함은 결국 lookup table을 만들 때, 몇개의 row를 구성해야 하는지를 알기 위해서
# # 라고 할 수 있지 않을까 생각된다.
#
# # [STUDY] Embedding layer의 param = vocab_size * embedding_dim ==> 즉 lookup table!!!!!
# #   output_shape = (input_length, embedding_dim)
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
#
# # metrics = acc => accuracy
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(X_train, y_train, epochs=100, verbose=2)
# model.summary()




