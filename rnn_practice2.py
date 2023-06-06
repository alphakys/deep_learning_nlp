import os
import warnings

from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

texts = """경마장에 있는 말이 뛰고 있다\n그의 말이 법이다\n가는 말이 고와야 오는 말이 곱다"""

tk = Tokenizer()
tk.fit_on_texts([texts])
vocab_size = len(tk.word_index) + 1

sequences = []
for line in texts.split('\n'):
    print(line)
    encoded = tk.texts_to_sequences([line])[0]

    for i in range(1, len(encoded)):
        sequence = encoded[:i + 1]
        sequences.append(sequence)

# print('sequences: ', sequences)

max_len = max([len(l) for l in sequences])
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

X = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocab_size)

from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN

# X의 argument를 보면 1 ~ 10까지의 index가 있기 때문에 input_dim을 10차원으로 설정한다.
embedding_dim = 10
hidden_units = 32

model = Sequential()
# [STUDY] Embedding layer에 대해서 더 공부 필요
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=200, verbose=2)

def sentence_gerneration(model, tk, current_word, n):
    init_word = current_word
    sentence = ''

    # n번 반복
    for _ in range(n):
        encoded = tk.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=5, padding='pre')
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tk.word_index.items():
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면 break
            if index == result:
                break

            # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' ' + word

        # 예측 단어를 문장에 저장
        sentence = sentence + ' ' + word


    sentence = init_word + sentence
    return sentence






