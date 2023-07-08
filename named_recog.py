import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/train.txt",
#     filename="train.txt")

f = open('train.txt')
tagged_sentences = []
sentence = []

for line in f:
    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
        if len(sentence) > 0:
            tagged_sentences.append(sentence)
            sentence = []
        continue
    splits = line.split(' ')  # 공백을 기준으로 속성을 구분한다.
    splits[-1] = re.sub(r'\n', '', splits[-1])  # 줄바꿈 표시 \n을 제거한다.
    word = splits[0].lower()  # 단어들은 소문자로 바꿔서 저장한다.
    sentence.append([word, splits[-1]])  # 단어와 개체명 태깅만 기록한다.

sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:  # 14,041개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence)  # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info))  # 각 샘플에서 개체명 태깅 정보만 저장한다.

# 단어 사이즈는 상위 4000개를 사용한다.
vocab_size = 4000
src_tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
src_tokenizer.fit_on_texts(sentences)

tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(ner_tags)

tag_size = len(tar_tokenizer.word_index) + 1

X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(ner_tags)

index_to_word = src_tokenizer.index_word
index_to_ner = tar_tokenizer.index_word

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=avr_len, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=8, validation_data=(X_test, y_test))

