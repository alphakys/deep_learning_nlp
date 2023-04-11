import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

raw_df = pd.read_csv("boston.csv")

ss = MinMaxScaler()  # StandardScaler
scaled_features = ss.fit_transform(raw_df[['RM', 'LSTAT']])


model = Sequential([
    Dense(1, input_shape=(2,), activation=None, kernel_initializer='zeros', bias_initializer='ones')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mse'])


#
model.fit(scaled_features, raw_df['PRICE'].values, epochs=1000, batch_size=30)


def get_batch_update_weight_values(x1, x2, target, w1, w2, bias, learning_rate=0.01):
    N = target.shape[0]

    predicted = w1 * x1 + w2 * x2 + bias
    diff = target - predicted
    # [STUDY] 1차원 리스트이면 각 자릿수끼리 곱해서 더한 값을 반환

    w1_update = (-2 / N) * learning_rate * np.dot(diff, x1)
    w2_update = (-2 / N) * learning_rate * np.dot(diff, x2)
    bias_update = (-2 / N) * learning_rate * (diff.sum())

    mse = np.mean(np.square(diff))

    return w1_update, w2_update, bias_update, mse


def batch_gradient_descent(scaled_features, target, verbose=False, iter_epochs=1000):
    learning_rate = 0.01

    w1 = np.zeros((1,), dtype=np.float64)
    w2 = np.zeros((1,), dtype=np.float64)
    bias = np.zeros((1,), dtype=np.float64)

    N = target.shape[0]
    batch_size = 30

    for i in range(iter_epochs):
        # batch_size만큼 랜덤하게 추출
        # choice의 최대 수를 N-30으로 해야 30개씩 랜덤하게 추출할 수 있다.
        # choice는 설정한 수보다 작은 수들에서 random하게 추출한다.
        container_num = np.random.choice(N - batch_size + 1, 1)[0]

        x1_batch = scaled_features[:, 0][container_num: container_num+30]
        x2_batch = scaled_features[:, 1][container_num: container_num+30]
        target_batch = target[container_num: container_num+30]

        w1_update, w2_update, bias_update, loss = get_batch_update_weight_values(x1_batch, x2_batch, target_batch, w1, w2, bias)

        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update

        if verbose:
            print(f'iter : {i + 1}')
            print(f'w1 {w1}, w2 {w2}, bias {bias} loss : {loss}')

    return w1, w2, bias

w1, w2, bias = batch_gradient_descent(scaled_features, raw_df['PRICE'].values, verbose=True, iter_epochs=1)



#################################################################


def get_sgd_update_weight_values(x1, x2, target, w1, w2, bias, learning_rate=0.01):
    N = target.shape[0]

    predicted = w1 * x1 + w2 * x2 + bias
    diff = target - predicted
    # [STUDY] 1차원 리스트이면 각 자릿수끼리 곱해서 더한 값을 반환

    w1_update = (-2 / N) * learning_rate * np.dot(diff, x1)
    w2_update = (-2 / N) * learning_rate * np.dot(diff, x2)
    bias_update = (-2 / N) * learning_rate * (diff.sum())

    mse = np.mean(np.square(diff))

    return w1_update, w2_update, bias_update, mse


def sgd_gradient_descent(scaled_features, target, verbose=False, iter_epochs=1000):
    learning_rate = 0.01

    w1 = np.zeros((1,), dtype=np.float64)
    w2 = np.zeros((1,), dtype=np.float64)
    bias = np.zeros((1,), dtype=np.float64)

    N = target.shape[0]

    for i in range(iter_epochs):

        stochastic_index = np.random.choice(N, 1)

        x1_sgd = scaled_features[:, 0][stochastic_index]
        x2_sgd = scaled_features[:, 1][stochastic_index]
        target_sgd = target[stochastic_index]

        w1_update, w2_update, bias_update, loss = get_sgd_update_weight_values(x1_sgd, x2_sgd, target_sgd, w1, w2, bias)

        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update

        if verbose:
            print(f'iter : {i + 1}')
            print(f'w1 {w1}, w2 {w2}, bias {bias} loss : {loss}')

    return w1, w2, bias

