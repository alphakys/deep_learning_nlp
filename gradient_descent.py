import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.linear_model import LinearRegression



# body_df = pd.DataFrame(
#     {"height": [65.78, 71.52, 69.40, 68.22, 67.79], "weight": [112.99, 136.49, 153.03, 142.34, 144.30]})

# print(body_df)
line_fitter = LinearRegression()

raw_df = pd.read_csv("boston.csv")
X = raw_df['LSTAT']
y = raw_df['PRICE']

# [STUDY]
#  1. reshape(a,b) a는 차원을 말함 // b는 a차원의 원소의 개수를 말함
#  x.values.reshape(1, -1)
linear_line = line_fitter.fit(X.values.reshape(-1, 1), y)

plt.plot(X, y, 'o')
plt.plot(X, line_fitter.predict(X.values.reshape(-1,1)), color='red')


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
