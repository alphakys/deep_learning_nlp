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


en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"

def load_preporcessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []
    with open('fra.txt', 'r') as lines:

        for i, line in enumerate(lines):
            # source 데이터와 target 데이터 분리
            if i < 10:
                split_data = line.split('\t')
                encoder_input.append(split_data[0])
                decoder_input.append('\t'+split_data[1])

    return encoder_input, decoder_input


'r'.strip()







