import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import re
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import unicodedata
import urllib3
from keras.layers import Embedding, GRU, Dense
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

# 33000개의 샘플을 사용한다.
num_samples = 33000


def to_ascii(s):
    # 프랑스어 악센트(accent) 삭제
    # 예시 : 'déjà diné' -> deja dine
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sent):
    # 악센트 제거 함수 호출
    sent = to_ascii(sent.lower())

    # 단어와 구두점 사이에 공백 추가.
    # ex) "I am a student." => "I am a student ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환.
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

    # 다수 개의 공백을 하나의 공백으로 치환
    sent = re.sub(r"\s+", " ", sent)
    return sent


def load_preporcessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []
    with open('fra.txt', 'r') as lines:

        for i, line in enumerate(lines):

            # source 데이터와 target 데이터 분리
            src_line, tar_line, _ = line.strip().split('\t')
            # source 데이터 전처리
            src_line = [w for w in preprocess_sentence(src_line).split()]
            # target 데이터 전처리
            tar_line = preprocess_sentence(tar_line)
            # start of signal을 삽인한 채 전처리 해준다.
            tar_line_in = [w for w in preprocess_sentence("<sos>" + tar_line).split()]
            # 나는 전 챕터에서 계속해서 의문이 들었다.
            # 왜 eos를 제대로 처리하지 않는걸까 이번 챕터에서 eos를 제대로 처리했다!
            tar_line_out = [w for w in preprocess_sentence(tar_line + "<eos>").split()]

            encoder_input.append(src_line)
            decoder_input.append(tar_line_in)
            decoder_target.append(tar_line_out)

            if i == num_samples - 1:
                break

    return encoder_input, decoder_input, decoder_target


# 전처리된 source 문장과 target 문장
encoded_sentences, sentences_fra_input, sentences_fra_output = load_preporcessed_data()

# encoding data padding
tk_en = Tokenizer(filters="", lower=False)
tk_en.fit_on_texts(encoded_sentences)
# 33000의 샘플과 영어 문장의 최대 길이는
encoder_input = tk_en.texts_to_sequences(encoded_sentences)
encoder_input = pad_sequences(encoder_input, padding='post')

# decoding data padding
tokenizer_fra = Tokenizer(filters="", lower=False)
tokenizer_fra.fit_on_texts(sentences_fra_input)
tokenizer_fra.fit_on_texts(sentences_fra_output)

decoder_input = tokenizer_fra.texts_to_sequences(sentences_fra_input)
decoder_input = pad_sequences(decoder_input, padding="post")

decoder_target = tokenizer_fra.texts_to_sequences(sentences_fra_output)
decoder_target = pad_sequences(decoder_target, padding="post")

src_vocab_size = len(tk_en.word_index) + 1
tar_vocab_size = len(tokenizer_fra.word_index) + 1

src_to_index = tk_en.word_index
index_to_src = tk_en.index_word
tar_to_index = tokenizer_fra.word_index
index_to_tar = tokenizer_fra.index_word

# 테스트 데이터를 섞어준다.
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

n_of_val = int(num_samples * 0.1)

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]








