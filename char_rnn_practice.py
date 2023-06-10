import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import tensorflow as tf

from string import punctuation

from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

from keras.utils import pad_sequences, to_categorical
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, TextVectorization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

f = open('11-0.txt', 'rb')
sentences_list = []

# text preprocessing
for sentence in f:
    sentence = sentence.strip()  # strip을 통해 \r, \n을 제거한다.
    sentence = sentence.lower()  # 소문자화
    sentence = sentence.decode('ascii', 'ignore')  # ascii 코드로 변환
    if len(sentence) > 0:
        sentences_list.append(sentence)
f.close()

# [STUDY] 하나 배움 .join()!!!!
raw_sentences = ' '.join(sentences_list)
# raw_sentences = ' '.join(sentences[:100])

test_t = sentences_list[:10]



tk = Tokenizer(filters='')
tk.fit_on_texts(raw_sentences)


# array([[   0,    0,    0, ...,    0,    0,  773],
#        [   0,    0,    0, ...,    0,  773,    4],
#        [   0,    0,    0, ...,  773,    4,    1],
#        ...,
#        [   0,    0,    0, ...,    0,  208,  537],
#        [   0,    0,    0, ...,  208,  537,  385],
#        [   0,    0,    0, ...,  537,  385, 2592]], dtype=int32)
# headline_list[:3]
# ['Rhythm of the Streets: ‘We’re Warrior Women, and Yes, We Can Play’',
# 'As Deficit Grows, Congress Keeps Spending', 'Lesson in Select Bus Service']
