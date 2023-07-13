import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

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

import tensorflow as tf
from keras.layers import Embedding, Input, TimeDistributed, Dropout, concatenate, Bidirectional, LSTM, Conv1D, Dense, MaxPooling1D, Flatten
from keras import Model
from keras.initializers.initializers_v2 import RandomUniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from seqeval.metrics import f1_score, classification_report
from keras_crf import CRFModel

# conv1d_out = TimeDistributed(Conv1D(filters=, kernel_size=, activation=))