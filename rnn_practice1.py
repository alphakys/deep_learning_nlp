import os

from numpy import tanh
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import SimpleRNN
from keras.models import Sequential

from keras.models import Model

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))

# model.summary()

# timesteps는 시점의 수입니다. 자연어 처리에서는 보통 문장의 길이입니다.
timesteps = 10
# 입력의 차원. 자연어 처리에서는 보통 단어 벡터의 차원
input_dim = 4
# hidden_units는 은닉 상태의 크기로 메모리 셀의 용량입니다.
# 초기 은닉 상태는 0의 값을 가지는 벡터로 초기화합니다.
hidden_units = 8

input_shape = (timesteps, input_dim)
# inputs는 (10, 4)
inputs = np.random.random((timesteps, input_dim))

# 초기의 hidden state를 zero로 초기화한다.
hidden_state_t = np.zeros((hidden_units,))

print(hidden_state_t)

Wx = np.random.random((hidden_units, input_dim))
Wh = np.random.random((hidden_units, hidden_units))


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# mnist 이미지 데이터를 가지고 생각한다면
# input_size = (28, 28)이다.

# LSTM cell의 개수는 28열에 대해서 처리할 수 있게 28개로 한다.
# 그리고 time step은 28개의 행을 처리해야 하기 때문에 28로 생각할 수있다.

# mnist의 행 개수
time_steps = 28

# 기억하고자 하는 정보(cell)의 개수 // 그리고 각 cell은 다음 cell에 영향을 미치고 동시에 자기 자신의 값을 기억한다.
num_units = 128

# 28개의 열[1행 -> 2행] 이렇게 한 번에 들어가는 것인가??
n_input = 28










