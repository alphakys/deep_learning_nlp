import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.text import Tokenizer
from pprint import pprint

tokenizer = Tokenizer()

texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']
# corpus list를 학습시킴
tokenizer.fit_on_texts(texts)

# 정수 인코딩
tokenizer.word_index

# [STUDY] 다만 주의할 점은 각 단어에 부여되는 인덱스는 1부터 시작하는 반면에 완성되는 행렬의 인덱스는 0부터 시작합니다.
#   실제로 단어의 개수는 9개였지만 완성된 행렬의 열의 개수는 10개인 것과 첫번째 열은 모든 행에서 값이 0인 것을 볼 수 있습니다.
#   인덱스 0에는 그 어떤 단어도 할당되지 않았기 때문입니다.

# 단어가 몇번 나왔는지도 체크함
tokenizer.texts_to_matrix(texts, mode='count')
# 단순히 단어가 존재했는지 그렇지 않은지만을 판단함
tokenizer.texts_to_matrix(texts, mode='binary')
# tfidf 행렬을 만듦
tokenizer.texts_to_matrix(texts, mode='tfidf').round(2)[1:]

# 각 문서에서의 각 단어의 등장 횟수를 분자로, 각 문서의 크기를 분모로 하는 표현 방법
tokenizer.texts_to_matrix(texts, mode='freq').round(2)[1:]


import pandas as pd

from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from keras.utils import to_categorical

newsdata = fetch_20newsgroups(subset='train')

print(newsdata.keys())
























