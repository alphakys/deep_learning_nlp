import numpy as np
import pandas as pd

from sklearn import datasets

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", nrows=4, skiprows=22,
                     header=None)  # , names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV", "PRICE"])

print(raw_df.head())

print('🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉')

print(raw_df.iloc[[0]])


exit()


# gradient descent() 함수에서 반복적으로 호출되면서 update될 weight/bias값을 계산하는 함수
# rm은 RM(방 개수), lstat(하위계층 비율), target은 price임. 전체 array가 다 입력됨.
# 반환 값은 weight와 bias가 update되어야 할 값가 mean squared error 값을 loss로 반환.
def get_update_weights_values(bias, w1, w2, rm, lstat, target, learning_rate):
    N = len(target)


# [STUDY]
#  1. df.index는 raw의 개수
#  2. df[key]는 key에 해당하는 column을 가져온다.
#  3.
# print(raw_df.index)


# 5  12


exit()

data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print(target)
