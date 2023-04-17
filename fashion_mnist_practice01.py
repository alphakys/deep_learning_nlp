import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import Input, Model
from keras.layers import Flatten, Dense

### 앞에서 생성한 로직들을 함수화
# * Functional API로 모델 만들기
# * pixel값 1 ~ 255를 0 ~ 1사이값 Float 32로 만들기
# * One Hot Encoding Label에 적용하기
# * 학습과 검증 데이터로 나누기.
# * compile, 학습/예측/평가

INPUT_SIZE = 28

import numpy as np


def create_model(INPUT_SIZE):
    # [STUDY] INPUT의 shape을 만들 때, 처리하고자 하는 데이터 한 행렬을 기준으로 만든다.
    #   차원이 아니라
    #   즉 fasion_mnist 모델을 예로든다면 60000개의 차원(6만개의 데이터세트)이 있고 각각이 28 by 28의 행렬이다.
    #   여기서 28 by 28의 행렬을 shape에 넣어줘야한다.

    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE))
    # flatten 해줄 때, input 값으로 call argument에 넣어준다.
    # flatten을 하면 1
    x = Flatten()(input_tensor)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=30, activation='relu')(x)
    output = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output, name='alpha_practice', )
    return model

