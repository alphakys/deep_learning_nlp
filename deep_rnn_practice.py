import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras import Sequential
from keras.optimizers import Adam
from keras.layers import SimpleRNN, Bidirectional
from numpy import tanh

import numpy as np

model = Sequential()

hidden_units = 10
timesteps = 10  # timesteps
input_dim = 5

model.add(SimpleRNN(units=hidden_units, input_length=timesteps, input_dim=input_dim, return_sequences=True))
model.add(SimpleRNN(units=hidden_units, return_sequences=True))


model2 = Sequential()
model2.add(Bidirectional(SimpleRNN(hidden_units, return_sequences=True), input_shape=(timesteps, input_dim)))

# 이유는 output_y는 하나이니까
# hidden_units의 개수만큼이 bias_h 였듯이
bias_output = 1
model2.summary()



















