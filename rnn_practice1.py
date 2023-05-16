import os

from numpy import tanh
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import SimpleRNN
from keras.models import Sequential

from keras.models import Model

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))

model.summary()


# 초기 은닉 상태 0(벡터)으로 초기화