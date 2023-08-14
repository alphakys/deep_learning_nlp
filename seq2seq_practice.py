import os
import shutil
import time
import zipfile

import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from keras.utils import to_categorical
import numpy as np
import timeit

lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
lines = lines[:60000]

lines.tar = lines.tar.apply(lambda x: '\t' + x + '\n')
src_vocab = set(''.join(lines['src'].values))
tar_vocab = set(''.join(lines['tar'].values))

# [STUDY] 중요!!! to_categorical을 해주니 len + 1을 해줌
src_vocab_size = len(src_vocab) + 1
tar_vocab_size = len(tar_vocab) + 1

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))

src_to_index = {v: i + 1 for i, v in enumerate(src_vocab)}
tar_to_index = {v: i + 1 for i, v in enumerate(tar_vocab)}

index_to_src = {v: k for k, v in src_to_index.items()}
index_to_tar = {v: k for k, v in tar_to_index.items()}

encoder_input = []

# 1개의 문장
for line in lines.src:
    encoded_line = []
    # 각 줄에서 1개의 char
    for char in line:
        # 각 char을 정수로 변환
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)

decoder_input = []
for line in lines.tar:
    encoded_line = []
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)

# 테스트에서는 <sos> -> \t를 입력으로 받고 그에 해당하는 출력을 다음 입력으로 사용하는 방식으로 테스트를 하지만
# 그렇다면 이 과정에서는 TimeDistributed(Dense(units=, activation='softmax))가 사용되겠지만
# train 과정에서는 ~ ~ ~ ~ <eos>를 label로 사용하도록 한다는 것 같다.
decoder_target = [decoder[1:] for decoder in decoder_input]

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])

encoder_input_pad = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input_pad = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target_pad = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

encoder_input = to_categorical(encoder_input_pad)
decoder_input = to_categorical(decoder_input_pad)
decoder_target = to_categorical(decoder_target_pad)

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Functional

# return_state=True이면 cell state와 hidden state까지 반환 받는다.
# return_sequences=True이면 매 step 마다 나오는 hidden state를 다 반환한다.

encoder_inputs = Input(shape=(None, src_vocab_size))
# return_state=True -> Last Hidden State + Last Hidden State + Last Cell State
# lstm output means finally Last Hidden State!!

# [STUDY] 중요!! return_sequences=True + return_state=True ==> All Hidden States + Last Hidden State + Last Cell State
encoder_lstm = LSTM(units=256, return_state=True)

# encoder_outputs은 여기서는 불필요
# 이유는 state_h랑 같기 때문에
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=1, validation_split=0.2)

# 1. return_state = True -> 1. last hidden state(encoder_outputs) 2. last hidden state 3. last cell state
# 2. return_sequences=True -> 1. all of hidden states
# 3. return_state=True, return_sequences=True -> 1. all of hidden states 2. last hidden state 3. last cell state

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태를 이전 시점의 상태로 사용.
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 lstm의 리턴하는 은닉 상태와 셀 상태를 버리지 않음
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())


def decode_sequence(input_seq):
    # get state from input
    # predict를 사용한다는 것은 대용량의 input에 대한 prediction을 얻기 위함이다.
    # encoder에서 나온 hidden_state와 cell_state가 -> state_value
    states_value = encoder_model.predict(input_seq)

    # sos에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1

    stop_condition = False
    decoded_sentence = ''

    # stop_condition이 True가 될 때까지 loop
    while not stop_condition:
        # 이전 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        # decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,)
        # sos 시그널이 들어갈 때는 endoeer에서 나온 encoder_states가 초기 값으로 들어감
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_tar_len:
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1,1,tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1

        states_value = [h, c]

    return decoded_sentence




















