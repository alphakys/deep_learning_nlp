import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import keras.initializers.initializers_v2
import numpy as np
import tensorflow as tf

from keras.layers import SimpleRNN, LSTM, Bidirectional
from keras.models import Sequential

train_X = np.array(
    [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]],
    dtype=np.float32)
train_X = np.expand_dims(train_X, axis=0)

# input_shape = (4, 5)
# batch_input_shape = (1, 4, 5)

# rnn = SimpleRNN(3, return_sequences=True)
# hidden_states = rnn(train_X)

# lstm = LSTM(3, return_sequences=True, return_state=True)
# hidden_state, last_state, last_cell_state = lstm(train_X)

k_init = tf.keras.initializers.Constant(value=0.1)
b_init = tf.keras.initializers.Constant(value=0)
r_init = tf.keras.initializers.Constant(value=0.3)

bilstm = Bidirectional(LSTM(3, return_sequences=True, return_state=True, kernel_initializer=k_init,
                            bias_initializer=b_init, recurrent_initializer=r_init))

hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(train_X)
