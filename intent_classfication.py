import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/11.%201D%20CNN%20Text%20Classification/dataset/intent_train_data.csv", filename="intent_train_data.csv")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/11.%201D%20CNN%20Text%20Classification/dataset/intent_test_data.csv", filename="intent_test_data.csv")

train_data = pd.read_csv('intent_train_data.csv')
test_data = pd.read_csv('intent_test_data.csv')

val_cnt = train_data['label'].value_counts()
val_cnt.plot(x=val_cnt.index, y=val_cnt.values, kind='bar')
plt.show()

intent_train = train_data['intent'].tolist()
label_train = train_data['label'].tolist()
intent_test = test_data['intent'].tolist()
label_test = test_data['label'].tolist()

idx_encode = preprocessing.LabelEncoder()
label_train = idx_encode.fit_transform(label_train)
label_test = idx_encode.fit_transform(label_test)

tk = Tokenizer()
tk.fit_on_texts(intent_train)
vocab_size = len(tk.word_index) + 1
threshold = 2
one_cnt_sum = 0
for w, cnt in tk.word_counts.items():
    if cnt < threshold:
        one_cnt_sum += 1

sequences = tk.texts_to_sequences(intent_train)
seq_cnt_list = [len(li) for li in sequences]
max_len = max(seq_cnt_list)
avg_len = sum(seq_cnt_list) / len(seq_cnt_list)

intent_train = pad_sequences(sequences, maxlen=max_len)
label_train = to_categorical(np.array(label_train))

indices = np.arange(intent_train.shape[0])

intent_train = intent_train[indices]
label_train = label_train[indices]

n_of_val = int(intent_train.shape[0] * 0.1)

X_train = intent_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = intent_train[-n_of_val:]
y_val = label_train[-n_of_val:]
X_test = intent_test
y_test = label_test

# train_test_split을 사용해도 된다.
# from sklearn.model_selection import train_test_split
# X_train, X_val = train_test_split(intent_train, test_size=0.1)

embedding_dict = {}
f = open('/media/alpha/Samsung_T5/deepLearning/glove.6B/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    raw_table = line.split()
    word = raw_table[0]
    embedding_dict[word] = np.array(raw_table[1:], dtype='float32')
f.close()

# glove의 embedding 차원이 100이기 때문에
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tk.word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras.models import Model
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate

kernel_size = [2, 3, 5]
num_filters = 512
dropout_ratio = 0.5
