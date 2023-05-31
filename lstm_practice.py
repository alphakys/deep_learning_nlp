import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf

from keras.layers import SimpleRNN, LSTM, Bidirectional
from keras.models import Sequential

train_X = np.array(
    [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]], dtype=np.float32)
train_X = np.expand_dims(train_X, axis=0)

# input_shape = (4, 5)
# batch_input_shape = (1, 4, 5)

rnn1 = SimpleRNN(3, return_sequences=True, return_state=True)
hidden_state1, last_state = rnn1(train_X)