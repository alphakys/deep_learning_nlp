import time
import timeit
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import nltk
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 토큰화에 품사 태깅이 된 데이터 받아오기
tagged_sentences = nltk.corpus.treebank.tagged_sents()

s = time.time()
sentences = [[t[0] for t in tag] for tag in tagged_sentences]
pos_tags = [[t[1] for t in tag] for tag in tagged_sentences]

e = time.time()
print(e-s)


s = time.time()
sentences, pos_tags = [], []
for tagged_sentence in tagged_sentences: # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어들은 sentence에 품사 태깅 정보들은 tag_info에 저장한다.
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(list(tag_info)) # 각 샘플에서 품사 태깅 정보만 저장한다.

e = time.time()
print(e-s)