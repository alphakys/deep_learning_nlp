import os
import shutil
import time
import zipfile

import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from keras.utils import to_categorical
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
from keras.models import Model
