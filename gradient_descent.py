import os
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

raw_df = pd.read_csv("boston.csv")
X = raw_df['LSTAT']
y = raw_df['PRICE']

# [STUDY]
#  용어정리
#  * NORMALIZATION / scikit-learn에서 MinMaxScaler를 사용하면 0과 1사이의 값으로 변환해준다.
#  * STANDARDIZATION / scikit-learn에서 StandardScaler를 사용하면 평균이 0이고 표준편차가 1인 값으로 변환해준다.
#  * np.array(X, float)는 C의 array와 똑같다고 한다. 그래서 두번째 argument에 type을 넣어주어야 함

# [STUDY]
#   StandardScaler 정리
#
#   data = [[0, 0], [0, 0], [1, 1], [1, 1]]
#   ssc.fit(data))
#   ssc.mean_ 평균
#   ssc.scale_ 표준편차
#   ssc.var_ 분산

ssc = StandardScaler()
x_reshape = X.values.reshape(-1, 1)


# 4.98
# 9.14
# 4.03
fit_x = ssc.fit(x_reshape)

print(fit_x.mean_)
print(fit_x.scale_)
print((4.98 - fit_x.mean_) / fit_x.scale_)

X_std = ssc.fit_transform(x_reshape)

print(X_std)
exit()

X_std1 = ssc.fit_transform(fitted_x)

print(X_std)
print('🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉')
print(X_std1)

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
