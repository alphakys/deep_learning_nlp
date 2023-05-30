import os

from numpy import tanh
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import SimpleRNN, Bidirectional
from keras.models import Sequential

from keras.models import Model

# [STUDY] return_sequences=True => 각 시점의 은닉 상태값을 반환한다.
#   return_sequences=False => 마지막 시점의 은닉 상태값만 반환한다.
model = Sequential()
model.add(SimpleRNN(units=3, batch_input_shape=(8, 2, 10), return_sequences=True))
model.summary()

timesteps = 10
input_dim = 4
hidden_units = 8

input_shape = (timesteps, input_dim)

inputs = np.random.random(input_shape)
hidden_state_t = np.zeros((hidden_units,))

print("초기 은닉 상태 : ", hidden_state_t)

# wx = np.random.random((input_dim, hidden_units))
# Wx = np.random.random((hidden_units, input_dim))

wx = np.ones((input_dim, hidden_units))
Wx = np.ones((hidden_units, input_dim))



wh = np.random.random((hidden_units, hidden_units))
bias_h = np.random.random((hidden_units,))

hidden_states_list = []
test_l = []

for input in inputs:
    output_t = np.tanh(np.dot(input, wx) + np.dot(hidden_state_t, wh) + bias_h)
    test_state = np.tanh(np.dot(Wx, input) + np.dot(wh, hidden_state_t) + bias_h)
    print(input)
    print(wx)
    print(np.dot(input, wx))
    print('🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉')

    print(input)
    print(Wx)
    print(np.dot(Wx, input))
    break



