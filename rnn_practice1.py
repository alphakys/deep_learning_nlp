import os

from numpy import tanh
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import SimpleRNN
from keras.models import Sequential

from keras.models import Model

# 메모리 셀의 최종 시점의 은닉 상태만을 리턴하고자 한다면 (batch_size, output_dim) 크기의 2D 텐서를 리턴합니다.
# model = Sequential()
# input_shape = [input_length, input_dim]과 같음
# 따라서 input_length => time_steps, input_dim => Dimensionality of the word representation

# 따라서 2D tensor임 => (hidden_units,(input_length, input_dim))
# model.add(SimpleRNN(units=3, input_shape=(2, 10)))

# model.summary()
# model1 = Sequential()
# model1.add(SimpleRNN(units=3, batch_input_shape=(8, 2, 10), return_sequences=True))

# 이미지로 이해하면 가로가 timesteps, 세로가 input_dim(word 개수) batch_size는 이미지의 개수(3차원 값)

# timesteps는 시점의 수입니다. 자연어 처리에서는 보통 문장의 길이입니다.
timesteps = 10
# 입력의 차원. 자연어 처리에서는 보통 단어 벡터의 차원
input_dim = 4
# hidden_units는 은닉 상태의 크기로 메모리 셀의 용량입니다.
# 초기 은닉 상태는 0의 값을 가지는 벡터로 초기화합니다.
hidden_units = 8

input_shape = (timesteps, input_dim)

# 입력 2D 텐서
# time_steps = 10은 문장의 길이
# input_dim은 4개의 단어

# 따라서
# 1. 나는1 너를 좋아해 그래
# 2. 나는 좋아해1 너를 그래
# 3. 좋아해 너를1 나는 그래
# 4. 나는1 너를 좋아해 그래
# 5. 나는 좋아해1 너를 그래
# 6. 좋아해 너를1 나는 그래
# 7. 나는1 너를 좋아해 그래
# 8. 나는 좋아해1 너를 그래
# 9. 좋아해 너를1 나는 그래
# 10. 나는1 너를 좋아해 그래

# 나는 좋아해 너를 나는 좋아해 너를 나는 좋아해 너를 나는

inputs = np.random.random(input_shape)

print(inputs.shape)
# 초기의 hidden state를 zero로 초기화한다.
hidden_state_t = np.zeros((hidden_units,))

Wx = np.random.random((hidden_units, input_dim))
Wh = np.random.random((hidden_units, hidden_units))
b = np.random.random((hidden_units,))

total_hidden_states = []

for idx, input_t in enumerate(inputs):
    print(f'단어 {idx} : {input_t}')
    print(f'단어 {idx} : {Wx}')


    print(Wx.shape, input_t.shape)
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    print(output_t)


# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()



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










