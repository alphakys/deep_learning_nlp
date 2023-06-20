import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten



from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# [STUDY] Embeddingd (number of samples, input_length) 이때 각 sample은 정수 인코딩이 된 결과로 정수 시퀀스입니다.
#   output으로 (batch_size, input_length, output_dim)의 3D 텐서를 리턴합니다.

sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality',
             'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]

tk = Tokenizer()
tk.fit_on_texts(sentences)
vocab_size = len(tk.word_index) + 1

embedding_dim = 8

X_encoded = tk.texts_to_sequences(sentences)
max_len = max([len(l) for l in X_encoded])
X_train = pad_sequences(X_encoded)
y_train = np.array(y_train)

model = Sequential()
# [STUDY] input_length -> time_steps --> 한 문장에서의 token의 길이
#   현재 학습하는 dataset에서는 각각의 리스트가 하나의 샘플로 구성된다.(batch)
# Embedding에서 vocab_size를 알고자 함은 결국 lookup table을 만들 때, 몇개의 row를 구성해야 하는지를 알기 위해서
# 라고 할 수 있지 않을까 생각된다.
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()



x = (7, 4)







