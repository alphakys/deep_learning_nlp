import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
# data = pd.read_csv("ner_dataset.csv", encoding="latin1")

data = pd.read_csv('ner_dataset.csv', encoding='latin1')

data = data.fillna(method="ffill")

data['Word'] = data['Word'].str.lower()

func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences = [t for t in data.groupby("Sentence #").apply(func)]

sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:  # 47,959개의 문장 샘플을 1개씩 불러온다.

    # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentence, tag_info = zip(*tagged_sentence)
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info))  # 각 샘플에서 개체명 태깅 정보만 저장한다.

len_list = [len(l) for l in sentences]
avr_size = sum(len_list) / len(len_list)
max_size = max(len_list)

src_tokenizer = Tokenizer(oov_token='OOV')
# 태깅 정보들은 내부적으로 대문자를 유지한 채 저장
tar_tokenizer = Tokenizer(lower=False)

src_tokenizer.fit_on_texts(sentences)
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1

X_data = src_tokenizer.texts_to_sequences(sentences)
y_data = tar_tokenizer.texts_to_sequences(ner_tags)

word_to_index = src_tokenizer.word_index
index_to_word = src_tokenizer.index_word
ner_to_index = tar_tokenizer.word_index
index_to_ner = tar_tokenizer.index_word
index_to_ner[0] = 'PAD'

decoded = []
for index in X_data[0] : # 첫번째 샘플 안의 인덱스들에 대해서
    decoded.append(index_to_word[index]) # 다시 단어로 변환

max_len = 70
X_data = pad_sequences(X_data, padding='post', maxlen=max_len)
y_data = pad_sequences(y_data, padding='post', maxlen=max_len)

X_train, X_test, y_train_int, y_test_int = train_test_split(X_data, y_data, test_size=.2, random_state=777)

y_train = to_categorical(y_train_int, num_classes=tag_size)
y_test = to_categorical(y_test_int, num_classes=tag_size)

import tensorflow as tf
from keras import Model
from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_crf import CRFModel
from seqeval.metrics import f1_score, classification_report

embedding_dim = 128
hidden_units = 64
dropout_ratio = 0.3

sequence_input = Input(shape=(max_len,), dtype=tf.int32, name='sequence_input')

model_embedding = Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=max_len)(sequence_input)

model_bilstm = Bidirectional(LSTM(units=hidden_units, return_sequences=True))(model_embedding)
# [STUDY] However, with return_sequences=False, Dense layer is applied only once at the last cell.
#   This is normally the case when RNNs are used for classification problem.
#   If return_sequences=True then Dense layer is applied to every timestep just like TimeDistributedDense.
model_dropout = TimeDistributed(Dropout(dropout_ratio))(model_bilstm)

model_dense = TimeDistributed(Dense(tag_size, activation='relu'))(model_dropout)

base = Model(inputs=sequence_input, outputs=model_dense)
model = CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics='accuracy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('bilstm_crf/cp.ckpt', monitor='val_decode_sequence_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

history = model.fit(X_train, y_train_int, batch_size=128, epochs=15, validation_split=0.1, callbacks=[es])

#
#
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))
# model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
# model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))
# model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, batch_size=128, epochs=6, validation_split=0.1)
for seq in test_seq:
    print([index_to_ner[entity] for entity in seq])