import os

from numpy import tanh
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import SimpleRNN, Bidirectional
from keras.models import Sequential

from keras.models import Model

# one layer rnn
model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))

model.summary()



# timesteps는 시점의 수입니다. 자연어 처리에서는 보통 문장의 길이입니다.
timesteps = 10
# 입력의 차원. 자연어 처리에서는 보통 단어 벡터의 차원
input_dim = 4
# hidden_units는 은닉 상태의 크기로 메모리 셀의 용량입니다.
# 초기 은닉 상태는 0의 값을 가지는 벡터로 초기화합니다.
hidden_units = 8

inputs = np.random.random((timesteps, input_dim))
hidden_states = np.zeros(hidden_units)

Wx_layer1 = np.random.random((hidden_units, input_dim))
Wh_layer1 = np.random.random((hidden_units, hidden_units))
bias_layer1 = np.random.random(hidden_units)

Wx_layer2 = np.random.random((hidden_units, input_dim))
Wh_layer2 = np.random.random((hidden_units, hidden_units))
bias_layer2 = np.random.random(hidden_units)

total_hidden_states_layer1 = []
total_hidden_states_layer2 = []

for input in inputs:

    calc_input = np.dot(Wx_layer1, input) + np.dot(Wh_layer1, hidden_states) + bias_layer1
    hidden_state_t = np.dot(Wx_layer1, input) + np.dot(Wh_layer1, hidden_states)
    output_t = np.tanh(calc_input)
    total_hidden_states_layer1.append(output_t)
    hidden_states = output_t

print(total_hidden_states_layer1)




# two layers rnn
model2 = Sequential()
model2.add(SimpleRNN(hidden_units, input_length=timesteps, input_dim=input_dim,
                    return_sequences=True))
model2.add(SimpleRNN(hidden_units, return_sequences=True))



# bidirectional rnn
input_dim2 = 5

model3 = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True),
                        input_shape=(timesteps, input_dim2)))