import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import SimpleRNN, Dense
from numpy import tanh

import pandas as df
from sklearn.preprocessing import MinMaxScaler


def get_train_test(url, split_percent=0.8):
    data_df = df.read_csv(url, usecols=[1], engine='python')
    data = np.array(data_df.values.astype('float32'))

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # data = scaler.fit_transform(data).flatten()
    n = len(data)
    split = int(n * split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data


sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data, test_data = get_train_test(sunspots_url)

# Prepare the input X and target Y
def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    # Prepare X
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y

time_steps = 12
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)



#
# hidden_units = 2
# dense_units = 1
#
# input_dim = 1
# time_steps = 3
#
# input_shape = (time_steps, input_dim)
#
# x_input = np.random.random(input_shape)
#
# wx = np.zeros((input_dim, hidden_units))
# wh = np.zeros((hidden_units, hidden_units))
# bh = np.zeros(hidden_units)
#
# print(wx.shape)
# print(wh.shape)
# print(bh.shape)
#
#
# def create_RNN(hidden_units, dense_units, input_shape, activation):
#     model = Sequential()
#     model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
#     model.add(Dense(dense_units, activation=activation[1]))
#     model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
#     return model
#
#
# demo_model = create_RNN(hidden_units, dense_units, (3, 1), activation=['linear', 'linear'])
#
# wx = demo_model.get_weights()[0]  # (1, 2)
# wh = demo_model.get_weights()[1]  # (2, 2)
# bh = demo_model.get_weights()[2]  # (2,)
# wy = demo_model.get_weights()[3]  # (2, 1)
# by = demo_model.get_weights()[4]  # (1,)
#
# x = np.array([1, 2, 3])
# x_input = np.reshape(x, (1, 3, 1))
# y_pred_model = demo_model.predict(x_input)
#
# m = 2
# h0 = np.zeros(m)
# h1 = np.dot(x[0], wx) + np.dot(h0, wh) + bh  # (1, 2)
# h2 = np.dot(x[1], wx) + np.dot(h1, wh) + bh  # (1, 2)
# h3 = np.dot(x[2], wx) + np.dot(h2, wh) + bh  # (1, 2)
# o3 = np.dot(h3, wy) + by
#
# print(h1)
# print(h2)
# print(h3)
# print(o3)
