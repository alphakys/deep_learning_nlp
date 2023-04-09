import math
import os
import random
import time
from decimal import Decimal
import numpy as np

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from numpy import ndarray
from pandas import Series

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

import matplotlib
matplotlib.use('TkAgg')


raw_df = pd.read_csv("boston.csv")
X1 = raw_df['RM']
X2 = raw_df['LSTAT']

# 생각해보자 price라고 하는 것은 결국 13개의 (혹은 더 많은 parameter가 있을 수 있겠지) feature에 의해서 도출된 결과값이다.
# 그리고 내가 지금까지 한 것은 한 feature에 대한 prediction을 수행한 것이었다.
# 여러 개의 feature에 대한 prediction을 수행할 수도 있었겠지

y = raw_df['PRICE'].values.reshape(-1, 1)

ss = StandardScaler()

st_X1: ndarray = ss.fit_transform(X=X1.values.reshape(-1, 1))
st_X2: ndarray = ss.fit_transform(X=X2.values.reshape(-1, 1))

# [STUDY]
#   (pred-true)**2 -> bias
#   (pred-pred_average)**2 -> variance(분산)
# 예측값과 (예측값들의 평균)의 차이가 얼마인지를 그리고 그 차이가 클수록 에러가 높다.
# irreducible error는 줄일 수 없는 에


# The data to fit
m = 20
theta1_true = 0.5
x = np.linspace(-1, 1, m)
y = theta1_true * x

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6.15))
ax[0].scatter(x, y, marker='x', s=40, color='k')

def cost_func(theta1):
    theta1 = np.atleast_2d(np.array(theta1))

    # [STUDY] 1차원과 2차원의 array를 얼마든지 행렬 계산 할 수 있다.
    #   굳이 1차원을 2차원으로 만들 필요가 없다.
    #   대신 1차원으로 값이 반환된다. 그리고 행렬 계산을 위한 행/렬의 개수는 맞아야 함
    # hypothesis.shape(50,20)
    # y.shape(20,)
    # 그래서 반환되는 값을 shape(50,20)을 row major order로 평균을 낸 값이 반횐됨

    # hypothesis(x, theta1) 이 값은 결국 x를 50세트를 만드는
    print(np.average((y - hypothesis(x, theta1)) ** 2, axis=1))
    print(np.average((y - hypothesis(x, theta1)) ** 2, axis=1) / 2)
    return np.average((y - hypothesis(x, theta1)) ** 2, axis=1) / 2
    #return np.average((y - hypothesis(x, theta1)) ** 2, axis=1) / 2


def hypothesis(x, theta1):
    # our hypothesis function, a straight line through the origin
    return theta1 * x

# theta1_grid가 x값
theta1_grid = np.linspace(-0.2, 1, 50)

#print(x)

J_grid = cost_func(theta1_grid.reshape(-1, 1))

J_grid = cost_func(theta1_grid)



exit()

# [STUDY]
#   1. StandardScaler한 input 데이터의 평균과 y의 평균을 구한다.
#   2. 그리고 true값이라고 해야할까. 여튼 평균을 기준으로 각 데이터들이 얼마나 떨어져 있는지를 구한다. 이것이 error이다.
#   3.

# [STUDY]
#   uniform안의 배열이 shape을 의미하는 듯하다.[3,3]
# weight
W = tf.Variable(tf.random.uniform([1], 0, 1.0))
# bias
b = tf.Variable(tf.random.uniform([1], 0, 1.0))

X = st_X1

H = W * X + b
cost = tf.reduce_mean(tf.square(H - y))

# 한 번에 얼마만큼 점프하는지 -> learning rate를 의미하는 듯
learning_rate = tf.Variable(0.01)

print(tf.GradientTape.gradient(3, 1))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)  # cost를 최소화하는 방향으로 train

init = tf.compat.v1.global_variables_initializer()

# [STUDY] 1은 row major order를 의미한다.
#    0은 column major order를 의미한다.


# cost = tf.reduce_mean()
# X = tf.p(tf.float32, shape=[None])


exit()


# print(residual_multiple_sum / residual_x_square.sum())
def ordinary_least_square(x: ndarray, y: ndarray):
    pass
    # residual_x = st_X1 - x_mean
    # residual_y = y - y_mean
    #
    # residual_multiple_sum = (residual_x * residual_y).sum()
    #
    # residual_x_square = residual_x ** 2
    #
    # ols_raw_list = (residual_x * residual_y) / residual_x_square
    # return np.sum(x*y)/np.sum(x*x)


# print((residual_x * residual_y)[:5])

exit()

lr1 = LinearRegression()
lr1.fit(st_X1, y)
lr1.predict(st_X1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6.15))

ax[0].scatter(st_X1, y, c='blue', marker='o', label='true')
ax[1].scatter(st_X1, lr1.predict(st_X1), c='k', marker='x', label='error')

fig.show()

exit()

lr1 = LinearRegression()
lr2 = LinearRegression()
# 이것은 결국 방개수에 대한 price를 예측하기 위한 그래프 즉 coefficient와 intercept를 구하기 위한 코드
lr1.fit(st_X1, y)
lr2.fit(st_X2, y)
# fit(coefficient, intercept)를 통해서 구해질 예측값


# ** 예측값들의 array **
prd_X1 = lr1.predict(st_X1)
prd_X2 = lr2.predict(st_X2)
# ** 실제값들의 array **
y_true = y.values

# m = 20
# # 가중치 인듯 정확하진 않음
# theta1_true = 0.5
# # x input data
# x = np.linspace(-1, 1, m)
# # true_y value
# y = theta1_true * x

# The plot: LHS is the data, RHS will be the cost function.
# [STUDY] 여러 개의 graph 축 공유
#   nrows 행으로 몇개의 그래프를 그릴 것인지, ncols 열로 몇개의 그래프를 그릴 것인지, fifsize는 그래프의 크기
#   var1, var2 하면 initialize하는 value가 array이면 0-index = var1, 1-index = var2
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6.15))
ax[0].scatter(st_X1, y, marker='x', s=30, color='k', label='Training data')


def cost_func(theta1):
    """The cost function, J(theta1) describing the goodness of fit."""
    theta1 = np.atleast_2d(np.asarray(theta1))
    return np.average((y - hypothesis(x, theta1)) ** 2, axis=1) / 2


def hypothesis(x, theta1):
    """Our "hypothesis function", a straight line through the origin."""
    return theta1 * x


# First construct a grid of theta1 parameter pairs and their corresponding
# cost function values.
theta1_grid = np.linspace(-0.2, 1, 50)
J_grid = cost_func(theta1_grid[:, np.newaxis])

# [STUDY] arr[:, np.newaxis]는 arr을 2차원으로 만들어준다. 즉 1차원은 2차원으로 / 2차원은 3차원으로 한 축을 늘려준다.
# The cost function as a function of its single parameter, theta1.
ax[1].plot(theta1_grid, J_grid, 'k')

print(J_grid)
# plt.show()

exit()

# Take N steps with learning rate alpha down the steepest gradient,
# starting at theta1 = 0.
N = 5
alpha = 1
theta1 = [0]
J = [cost_func(theta1[0])[0]]
for j in range(N - 1):
    last_theta1 = theta1[-1]
    this_theta1 = last_theta1 - alpha / m * np.sum(
        (hypothesis(x, last_theta1) - y) * x)
    theta1.append(this_theta1)
    J.append(cost_func(this_theta1))

# Annotate the cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
# Also plot the fit function on the LHS data plot in a matching colour.
colors = ['b', 'g', 'm', 'c', 'orange']
ax[0].plot(x, hypothesis(x, theta1[0]), color=colors[0], lw=2,
           label=r'$\theta_1 = {:.3f}$'.format(theta1[0]))
for j in range(1, N):
    ax[1].annotate('', xy=(theta1[j], J[j]), xytext=(theta1[j - 1], J[j - 1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
    ax[0].plot(x, hypothesis(x, theta1[j]), color=colors[j], lw=2,
               label=r'$\theta_1 = {:.3f}$'.format(theta1[j]))

# Labels, titles and a legend.
ax[1].scatter(theta1, J, c=colors, s=40, lw=0)
ax[1].set_xlim(-0.2, 1)
ax[1].set_xlabel(r'$\theta_1$')
ax[1].set_ylabel(r'$J(\theta_1)$')
ax[1].set_title('Cost function')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Data and fit')
ax[0].legend(loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()

exit()

a = np.linspace(1, 10, num=10)

print(a)
exit()
print("mse : ", mse(y_true, prd_X1, multioutput='raw_values'))

exit()

# [STUDY] lr.score() 결정계수라고 하며
#   예측한 값과 TRUE 값의 차이가 얼마나 적냐 즉 정확도가 얼마나 높냐를 나타내는 지표이다.
#   0~ 1 사이의 값을 가지며 1에 가까울수록 정확도가 높다.
print('x1 score : ', lr1.score(st_X1, y_true))
print('x2 score : ', lr2.score(st_X2, y_true))

# 기존 데이터를 바탕으로 도출된 그래프
# plt.plot(X1, y, 'o')
# 내가 학습시킨(coefficient, intercept)를 구한 모델을 바탕으로 그려진 그래프
plt.plot(st_X1, lr1.predict(st_X1), color='black')
plt.plot(st_X2, lr2.predict(st_X2), color='blue')

plt.show()

exit()

# [STUDY]
#   line_fitter.coef는 기울기를 의미한다.
#   line_fitter.intercept는 y절편을 의미한다.


# [STUDY] ROW MAJOR ORDER -> ------->
#  COLUMN MAJOR ORDER -> ||||||||||

raw_df = pd.read_csv("boston.csv")
X1 = raw_df['RM']
X2 = raw_df['LSTAT']
y = raw_df['PRICE']

multi_dataframe = raw_df[[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]

# [STUDY]
#  scaling
#  MinMaxScaler란 데이터의 최대값과 최소값을 바탕으로 모든 데이터를 0에서 1사이의 값으로 환산하는 것을 의미한다.

ms = MinMaxScaler()
two_feature_dataframe = raw_df[['RM', 'LSTAT']]

plt.scatter(two_feature_dataframe['RM'].values, two_feature_dataframe['LSTAT'].values, alpha=0.4)

plt.show()

exit()
# [STUDY]
#  dataframe.values를 사용하여 데이터만 전달되도록 한다.
x_train, x_test, y_train, y_test = train_test_split(ms_fit_X, y, train_size=0.85, test_size=0.15)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

score = mlr.score(x_train, y_train)
print(score)

# linear regressoin에 train 데이터를 학습한 모델을 바탕으로 test 데이터를 예측한 값
y_predict = mlr.predict(x_test)

# linear regressoin에 train 데이터를 학습한 모델을 바탕으로 test 데이터를 예측한 값
# y_test는 실제 방값
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

exit()
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


ms = MinMaxScaler()
scale_X1 = ms.fit_transform(X1.values.reshape(-1, 1))
scale_X2 = ms.fit_transform(X2.values.reshape(-1, 1))

lr = LinearRegression()
line_fitter = lr.fit(scale_X1, y)

plt.plot(scale_X1, y, 'o')
plt.plot(scale_X1, lr.predict(scale_X1), color='red')

plt.show()

# print(scale_X)

# plt.plot(X, y, 'o')
# plt.plot(X, line_fitter.predict(X.values.reshape(-1, 1)), color='red')


exit()

ssc = StandardScaler()

x_reshape = X.values.reshape(-1, 1)

fit_x = ssc.fit(x_reshape)

X_std = ssc.fit_transform(x_reshape)

print(X_std)

exit()

# StandardScaler().fit_transform(X)


exit()
line_fitter = LinearRegression()

# [STUDY]
#  1. reshape(a,b) a는 차원을 말함 // b는 a차원의 원소의 개수를 말함
#  x.values.reshape(1, -1)
#  fit 메서드는 기울기 line_fitter.coef_와 절편 line_fitter.intercept_를 전달한다.
#  여기서 주의해야 할 점은 X데이터를 넣을 때 .values.reshape(-1,1)를 해줬다는 거다. 왜냐하면 X는 2차원 array 형태여야 하기 때문이다.
#  이런 식으로 [[x1], [x2], [x3], ... , [xn]] . (이렇게 넣는 이유는 X 변수가 하나가 아니라 여러개일 때 다중회귀분석을 실시하기 위함인데,
#  이는 다른 포스팅에서 소개한다.)
linear_line = line_fitter.fit(X.values.reshape(-1, 1), y)

print(linear_line.predict([[20]]))

print(linear_line.intercept_)

plt.plot(X, y, 'o')
plt.plot(X, line_fitter.predict(X.values.reshape(-1, 1)), color='red')

plt.show()

exit()
raw_df = pd.read_csv("boston.csv")

# [STUDY]
#  6.3200e-03 -> 부동소수점으로 e-03은 10의 -3승을 의미한다.
print(raw_df)

exit()


# gradient descent() 함수에서 반복적으로 호출되면서 update될 weight/bias값을 계산하는 함수
# rm은 RM(방 개수), lstat(하위계층 비율), target은 price임. 전체 array가 다 입력됨.
# 반환 값은 weight와 bias가 update되어야 할 값과 mean squared error 값을 loss로 반환.
def get_update_weights_values(bias, w1, w2, rm, lstat, target, learning_rate):
    N = len(target)
    predicted = w1 * rm + w2 * lstat + bias
    diff = target - predicted


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

exit()


def create_boston_dataset():
    # [STUDY]
    #  boston housing dataset csv 새로 생성
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # [STUDY]
    #  1. l[:3]은 0,1,2번째( X < 3 ) index를 가져온다.
    #  2. l[x:y]는 x <= index < y를 가져온다.
    #  3. l[x:y:z]는 x <= index < y를 z만큼 건너뛰면서 가져온다.
    #  4. iloc는 (index) location을 의미하는 듯 -> 따라서 row(index)에 해당하는 한 / [가로줄]을 가져온다.
    #  5. df.iloc[:, 0]은 df의 첫번째 column의 열을 가져온다. / [세로줄]
    #  6. [] slicing에 , 뒤로는 열[column, 세로줄] // 앞으로는 행[row, 가로줄]을 의미한다.
    #  7.

    end = len(raw_df.columns)
    end_row = raw_df.index.stop

    ins_series: Series = raw_df.iloc[1::2, 0:end_row]
    ins_series.reset_index(inplace=True, drop=True)

    raw_df.drop(index=range(0, end_row)[1::2], axis=0, inplace=True)
    raw_df.reset_index(inplace=True, drop=True)

    for i in range(end, end + 3):
        raw_df[i] = ins_series[i - 11]

    raw_df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
                      "PRICE"]
