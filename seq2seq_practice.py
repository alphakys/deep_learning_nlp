import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from keras.utils import pad_sequences
from keras.utils import to_categorical

filename = 'fra-eng.zip'
basepath = os.getcwd()
filepath = os.path.join(basepath, filename)

with zipfile.ZipFile(filepath, mode='r') as zipfile:
    zipfile.extractall(basepath)

lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']

lines = lines[0:60000]
lines['tar'] = lines['tar'].apply(lambda x: '\t' + x + '\n')

src_vocab = set()
for line in lines.src:  # 1줄씩 읽음
    for char in line:  # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab = set()
for line in lines.src:  # 1줄씩 읽음
    for char in line:  # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab) + 1
tar_vocab_size = len(tar_vocab) + 1

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))

src_to_index = dict([(word, i + 1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i + 1) for i, word in enumerate(tar_vocab)])

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

decoder_target = []
for line in lines.tar:
    timestep = 0
    encoded_line = []
    for char in line:
        if timestep > 0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

# [STUDY] 의문점 해결
#   Input layer의 shape에 None을 입력해도 알아서 인식한다. 결국 가장 마지막 dimension 값을 넣어주는 것이 중요하다.


# [STUDY] return_sequences와 return_state의 설명을 알고 싶다면
#  https://wikidocs.net/106473
from keras.models import Model
from keras.layers import Input, LSTM
import numpy as np
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)

# encoder_outputs은 여기서는 불필요
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c]



