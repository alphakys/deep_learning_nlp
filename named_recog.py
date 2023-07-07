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
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
        if len(sentence) > 0:
            tagged_sentences.append(sentence)
            sentence = []
        continue
    splits = line.split(' ') # 공백을 기준으로 속성을 구분한다.
    splits[-1] = re.sub(r'\n', '', splits[-1]) # 줄바꿈 표시 \n을 제거한다.
    word = splits[0].lower() # 단어들은 소문자로 바꿔서 저장한다.
    sentence.append([word, splits[-1]]) # 단어와 개체명 태깅만 기록한다.

tagged_sentences
for sen in tagged_sentences:
    tmp_sen = list(zip(*sen))

sentences, ner_tags = [], []

for sen in tagged_sentences:
    tmp_sen = list(zip(*sen))
    sentences.append(tmp_sen[0])
    ner_tags.append(tmp_sen[1])

sentences[0]
sentences[3]
ner_tags[2]
len(sentences)
len(ner_tags)
len_list = [len(sen) for sen in sentences]
ner_tk = Tokenizer()
sentences, ner_tags = [], []

for sen in tagged_sentences:
    tmp_sen = list(zip(*sen))
    sentences.append(list(tmp_sen[0]))
    ner_tags.append(list(tmp_sen[1]))

del sentence_tk
sentence_tk = Tokenizer()
sentence_tk.fit_on_texts(sentences)
sentence_tk.word_index
vocab_size = len(sentence_tk.word_index) + 1
vocab_size
len_list
max(len_list)
sum(len_list) / len(len_list)
int(sum(len_list) / len(len_list))
avr_len = 14
plt.hist(len_list, bins=50)
plt.show()
sentences[1]
sen_tokens = sentence_tk.texts_to_sequences(sentences)
sen_tokens[1]
sen_tokens[3]
padded_sequences = pad_sequences(sen_tokens)
padded_sequences = pad_sequences(sen_tokens, maxlen=avr_len)
padded_sequences
padded_sequences.shape
ner_tk.fit_on_texts(ner_tags)
ner_tk.index_word
ner_tags[1]
ner_tags[3]
tag_size = len(ner_tk.index_word) + 1
ner_tokens = ner_tk.texts_to_sequences(ner_tags)
ner_tokens[1]
padded_sequences = pad_sequences(sen_tokens, maxlen=avr_len, padding='post')
padded_sequences[3]
padded_sequences[4]
padded_sequences[5]
padded_sequences[10]
padded_sequences[11]
padded_sequences[12]
ner_tokens
padded_labels = pad_sequences(ner_tokens, maxlen=avr_len)
padded_labels
processed_labels = to_categorical(ner_tokens)
processed_labels = to_categorical(padded_labels)
processed_labels
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, processed_labels, test_size=0.25,
                                                    random_state=1004)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, processed_labels, test_size=0.25,
                                                    random_state=1004)
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam

embedding_dim = 128
hidden_units = 128
model.add(Embedding(vocab_size, embedding_dim, input_length=avr_len))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, activation='relu')))
del model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=avr_len, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=8, validation_data=(X_test, y_test))
X_test[0]
model.predict(X_test[1])
X_test
X_test.shape
X_test[1]
model.summary()
tag_size