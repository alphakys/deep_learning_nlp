import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import SimpleRNN, Dense
from numpy import tanh

hidden_units = 2
dense_units = 1
input_shape = (3, 1)

time_steps = 3
input_dim = 1


wx = np.zeros((input_dim, hidden_units))
wh = np.zeros((hidden_units, hidden_units))
b = np.zeros(hidden_units)

print(wx.shape)
print(wh.shape)
print(b.shape)

# time_steps, input_length =3
# input_dim = 1

def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

    return model


demo_model = create_RNN(hidden_units, dense_units, input_shape, activation=['linear', 'linear'])

wx = demo_model.get_weights()[0]
wh = demo_model.get_weights()[1]
bh = demo_model.get_weights()[2]
wy = demo_model.get_weights()[3]
by = demo_model.get_weights()[4]

print(wx.shape)
print(wh.shape)
print(b.shape)
print(wy.shape)
print(by.shape)
x = np.array([1, 2, 3])
x_input = np.reshape(x, (1, 3, 1))
y_pred_model = demo_model.predict(x_input)

m = 2
h0 = np.zeros(m)
h1 = x[0]*wx

























ht = np.array([0, 0, 0])
