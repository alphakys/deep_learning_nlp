import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

raw_df = pd.read_csv("boston.csv")

ss = MinMaxScaler()  # StandardScaler
scaled_features = ss.fit_transform(raw_df[['RM', 'LSTAT']])


def get_update_weight_values(x1, x2, target, learning_rate, w1, w2, bias):
    N = len(target)
    predicted = w1 * x1 + w2 * x2 + bias

    diff = target - predicted
    # [STUDY] 1차원 리스트이면 각 자릿수끼리 곱해서 더한 값을 반환

    w1_update = (-2 / N) * learning_rate * np.dot(diff, x1)
    w2_update = (-2 / N) * learning_rate * np.dot(diff, x2)
    bias_update = (-2 / N) * learning_rate * (diff.sum())

    mse = np.mean(np.square(diff))

    return w1_update, w2_update, bias_update, mse

def gradient_descent(scaled_features, target, verbose=False, iter_epochs=1000):

    learning_rate = 0.01
    x1 = scaled_features[:, 0]
    x2 = scaled_features[:, 1]

    w1 = np.zeros((1,), dtype=np.float64)
    w2 = np.zeros((1,), dtype=np.float64)
    bias = np.zeros((1,), dtype=np.float64)

    for i in range(iter_epochs):
        print(f'iter : {i+1}')
        w1_update, w2_update, bias_update, loss = get_update_weight_values(x1, x2, target, learning_rate, w1, w2, bias)

        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update

        print(f'w1 {w1}, w2 {w2}, bias {bias} loss : {loss}')

gradient_descent(scaled_features, raw_df['PRICE'].values, verbose=True)
