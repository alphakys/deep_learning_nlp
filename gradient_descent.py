import os
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from numpy import ndarray
from pandas import Series

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import load_diabetes

import matplotlib

# matplotlib.use('TkAgg')


raw_df = pd.read_csv("boston.csv")

ss = StandardScaler()
standard_fit_input = ss.fit_transform(X=raw_df[['RM', 'LSTAT']])
y = raw_df['PRICE'].values


# weight와 bias 값을 조정하면서 최적의 cost(loss)를 찾아가는 것이다.
# 이 과정에서 weight의 update가 일어나는데 편미분을 이용하여 weight의 변화율(기울기)를 구하고
# 기울기의 양수, 음수에 따라서 그 weight 값을 증가시키거나 감소시키는 것이다.
# 기울기와 반대로 weight를 조정하는 것이기 때문에 w_new = w_old - learning_rate * gradient이다.


# [STUDY] 중요!! weight update 공식
#  w_new = w_old - (-2 * (target - prediction) * x)
#  bias_new = bias_old - (-2 * (target - prediction) * 1)
def get_update_weights_value(bias, w1, w2, rm, lstat, target, learning_rate=0.01):
    # 데이터 건수
    N = len(target)
    # 예측 값.
    predicted = w1 * rm + w2 * lstat + bias
    # 실제값과 예측값의 차이
    diff = target - predicted

    # bias 를 array 기반으로 구하기 위해서 설정.
    bias_factors = np.ones((N,))

    # weight와 bias를 얼마나 update할 것인지를 계산.
    w1_update = -(2 / N) * learning_rate * (np.dot(rm.T, diff))
    w2_update = -(2 / N) * learning_rate * (np.dot(lstat.T, diff))
    bias_update = -(2 / N) * learning_rate * (np.dot(bias_factors.T, diff))

    # Mean Squared Error값을 계산.
    mse_loss = np.mean(np.square(diff))

    # weight와 bias가 update되어야 할 값과 Mean Squared Error 값을 반환.
    return bias_update, w1_update, w2_update, mse_loss


def gradient_descent(features, target, iter_epochs=1000, verbose=True):
    # w1, w2는 numpy array 연산을 위해 1차원 array로 변환하되 초기 값은 0으로 설정
    # bias도 1차원 array로 변환하되 초기 값은 1로 설정.
    w1 = np.zeros((1,))
    w2 = np.zeros((1,))
    bias = np.zeros((1,))
    print('최초 w1, w2, bias:', w1, w2, bias)

    # learning_rate와 RM, LSTAT 피처 지정. 호출 시 numpy array형태로 RM과 LSTAT으로 된 2차원 feature가 입력됨.
    learning_rate = 0.01
    rm = features[:, 0]
    lstat = features[:, 1]

    # iter_epochs 수만큼 반복하면서 weight와 bias update 수행.
    for i in range(iter_epochs):
        # weight/bias update 값 계산
        bias_update, w1_update, w2_update, loss = get_update_weights_value(bias, w1, w2, rm, lstat, target,
                                                                           learning_rate)
        # weight/bias의 update 적용.
        w1 = w1 - w1_update
        w2 = w2 - w2_update
        bias = bias - bias_update
        if verbose:
            print('Epoch:', i + 1, '/', iter_epochs)
            print('w1:', w1, 'w2:', w2, 'bias:', bias, 'loss:', loss)

    return w1, w2, bias


scaler = MinMaxScaler()
# pandas loc를 이용하여 exclude 가능
# scaled_features = scaler.fit_transform(raw_df.loc[:, raw_df.columns != 'PRICE'])
scaled_features = scaler.fit_transform(raw_df[['RM', 'LSTAT']])
gradient_descent(scaled_features, y, verbose=True)


# [STUDY] slicing시 행렬에 맞게 출력됨
# print(raw_df[:509])
# 5행, 3열까지 // 13열까지 있음
# print(raw_df.iloc[:5,0])

model = Sequential([
    # 단 하나의 units 설정, input shape는 2차원, 회귀이므로 activation은 설정하지 않음
    # weight와 bias 초기화는 kernel_inbitializer와 bias_initializer로 설정
    # 케라스에서는 shape(행, 렬)이 반대인 듯하다.
    Dense(units=1, input_shape=(13,), kernel_initializer='zeros', bias_initializer='ones'),
])

# Adam optimizer를 이용하여 loss 함수는 Mean squared error, 성능 측정 역시 MSE를 이용하여 학습 수행
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mse'])
model.fit(scaled_features, y, epochs=1000)

keras_prd = model.predict(scaled_features)
raw_df['keras_prd'] = keras_prd
print(raw_df.head(15))
print()

exit()

scaled_features = standard_fit_input
w1, w2, bias = gradient_descent(scaled_features, y, iter_epochs=5000, verbose=False)

predicted = w1 * scaled_features[:, 0] + w2 * scaled_features[:, 1] + bias


# [STUDY]
#   1. StandardScaler한 input 데이터의 평균과 y의 평균을 구한다.
#   2. 그리고 true값이라고 해야할까. 여튼 평균을 기준으로 각 데이터들이 얼마나 떨어져 있는지를 구한다. 이것이 error이다.
#   3.

# [STUDY]
#   uniform안의 배열이 shape을 의미하는 듯하다.[3,3]


# [STUDY] 1은 row major order를 의미한다.
#    0은 column major order를 의미한다.


# [STUDY] 여러 개의 graph 축 공유
#   nrows 행으로 몇개의 그래프를 그릴 것인지, ncols 열로 몇개의 그래프를 그릴 것인지, fifsize는 그래프의 크기
#   var1, var2 하면 initialize하는 value가 array이면 0-index = var1, 1-index = var2


# [STUDY] lr.score() 결정계수라고 하며
#   예측한 값과 TRUE 값의 차이가 얼마나 적냐 즉 정확도가 얼마나 높냐를 나타내는 지표이다.
#   0~ 1 사이의 값을 가지며 1에 가까울수록 정확도가 높다.


# [STUDY]
#   line_fitter.coef는 기울기를 의미한다.
#   line_fitter.intercept는 y절편을 의미한다.


# [STUDY] ROW MAJOR ORDER -> ------->
#  COLUMN MAJOR ORDER -> ||||||||||

# [STUDY]
#  scaling
#  MinMaxScaler란 데이터의 최대값과 최소값을 바탕으로 모든 데이터를 0에서 1사이의 값으로 환산하는 것을 의미한다.

# [STUDY]
#  dataframe.values를 사용하여 데이터만 전달되도록 한다.


# [STUDY]
#  용어정리
#  * NORMALIZATION / scikit-learn에서 MinMaxScaler를 사용하면 0과 1사이의 값으로 변환해준다.
#    NORMALIZATION 공식 = (x(테스트 값) - min) / (max - min)
#    결국엔 분자에 최대값 보다 작거나 같은 값들이 들어가니까 1보다 큰 수가 나올 수가 없다. 이해 완료
#  * STANDARDIZATION / scikit-learn에서 StandardScaler를 사용하면 평균이 0이고 표준편차가 1인 값으로 변환해준다.
#    STANDARDIZATION 공식 = z = (x(테스트 값) - u(평균)) / s(표준편차)
#  * np.array(X, float)는 C의 array와 똑같다고 한다. 그래서 두번째 argument에 type을 넣어주어야 함

# [STUDY]
#   StandardScaler 정리
#   data = [[0, 0], [0, 0], [1, 1], [1, 1]]
#   ssc.fit(data))
#   ssc.mean_ 평균
#   ssc.scale_ 표준편차
#   ssc.var_ 분산

# [STUDY]
#  스케일 하는 방법
#   scale_X1 = ms.fit_transform(X1.values.reshape(-1, 1))
#   scale_X2 = ms.fit_transform(X2.values.reshape(-1, 1))


# [STUDY]
#  1. reshape(a,b) a는 차원을 말함 // b는 a차원의 원소의 개수를 말함
#  x.values.reshape(1, -1)
#  fit 메서드는 기울기 line_fitter.coef_와 절편 line_fitter.intercept_를 전달한다.
#  여기서 주의해야 할 점은 X데이터를 넣을 때 .values.reshape(-1,1)를 해줬다는 거다. 왜냐하면 X는 2차원 array 형태여야 하기 때문이다.
#  이런 식으로 [[x1], [x2], [x3], ... , [xn]] . (이렇게 넣는 이유는 X 변수가 하나가 아니라 여러개일 때 다중회귀분석을 실시하기 위함인데,
#  이는 다른 포스팅에서 소개한다.)


# [STUDY]
#  1. df.index는 raw의 개수
#  2. df[key]는 key에 해당하는 column을 가져온다.



ss = StandardScaler()
scaled_features = ss.fit_transform(raw_df[['RM', 'LSTAT']])
print(scaled_features)
exit()


def get_update_weight_values():
    pass



def gradient_descent(scaled_features, target, verbose=False, iter_epochs=1000):

    learning_rate = 0.01




















