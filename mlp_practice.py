import os
import warnings

from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from keras import Input

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import TextVectorization

from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.optimizers import Adam

newsdata = fetch_20newsgroups(subset='train')
news_df = pd.DataFrame(data=newsdata.data, columns=['email'])
news_df['target'] = pd.Series(newsdata.target)

news_test = fetch_20newsgroups(subset='test', shuffle=True)
train_email = news_df['email']
train_label = news_df['target']

test_email = news_test.data
test_label = news_test.target

# 데이터의 최대 단어 수를 정의하는 파라미터
vocab_size = 10000
# 분류하고자 하는 뉴스 카테고리의 수 => target의 수
num_classes = 20


def prepare_data(train_data, test_data, mode):
    # vocab 사이즈 만큼의 단어만 사용한다.
    # num_words – the maximum number of words to keep, based on word frequency.
    # Only the most common `num_words-1` words will be kept.

    # 의문점! 만약에 train_dataset만 tokenizing을 한다면 test dataset에는 없는 단어가 있을 수 있다.
    # 이 부분에 대해서 어떻게 처리해줄 것인가에 대해서 고민해야 하지 않나?
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train_data)
    # 행렬에는 빈도수 기준 상위 9,999개의 단어가 표현된 셈입니다. 빈도수 상위 1번 단어와 9,999번 단어를 확인해보겠습니다.
    # fit_on_textxs를 통해서 단어의 빈도수를 기준으로 indexing을 하는 듯하다.

    X_train = tokenizer.texts_to_matrix(train_data, mode=mode)
    X_test = tokenizer.texts_to_matrix(test_data, mode=mode)

    return X_train, X_test, tokenizer.word_index


X_train, X_test, word_to_index = prepare_data(train_email, test_email, 'binary')

train_label = to_categorical(train_label, num_classes=num_classes)
test_label = to_categorical(test_label, num_classes=num_classes)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def fit_and_evaluate(train_data, test_data, train_label, test_label):
    model = Sequential()
    # input_tensor = Input(shape=(train_data.shape))
    # [STUDY] Dense layer =>
    #   units: 출력 뉴런의 수를 결정한다.
    #   input_dim : 입력 뉴런의 수를 설정한다.
    #   kernel_initilizer : 가중치를 초기화하는 방법 설정
    #       uniform : 균등분포
    #       normal : 가우시안 분포
    #   activation : 활성화 함수 설정

    # [STUDY] 이진분류 문제는 그렇다면 출력충이 하나의 값 => 0 / 1의 값으로만 반환되면 되기 때문에
    #   출력층의 개수는 1개, 입력값의 수는 3개 일때, 그리고 활성화 함수는 시그모이드 함수 이유는 이진분류이기 때문
    #   Dense(units=1, input_dim=3, activation='sigmoid')
    #   다중클래스 분류문제에서 입력값이 4개, 출력값이 3개인 경우, 출력층으로 사용된 Dense 레이어
    #   Dense(units=3, input_dim=4, activation='softmax')

    # [STUDY] !! 핵심 !!
    #  입력층이 아닐 때는 결국 그 이전 층의 units가 입력값이 되기 때문에 모든 Dense layer에서 input_dim을 설정할 필요가 없다.
    model.add(Dense(units=256, input_shape=train_data.shape))


# input_dim = 4, features = 10
# input_shape = (4, 10)
# dense1 = Dense(units=3, input_dim=4, activation='sigmoid')
# dense2 = Dense(units=5, activation='sigmoid')


(2, 4)

    n1
x1
    n2

    n3
x2
    n4









