import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from matplotlib import pyplot as plt


# x = wx + b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x * 10)

plt.plot(x, y, 'g')
# plt.show()


# 이렇게 뉴런에서 출력값을 변경시키는 함수를 활성화 함수(Activation Function)라고 합니다.
# 초기 인공 신경망 모델인 퍼셉트론은 활성화 함수로 계단 함수를 사용하였지만,
# 그 뒤에 등장한 여러가지 발전된 신경망들은 계단 함수 외에도 여러 다양한 활성화 함수를 사용하기 시작했습니다.
# 사실 앞서 배운 시그모이드 함수나 소프트맥스 함수 또한 활성화 함수 중 하나입니다.
def and_gate(x1, x2):
    w1 = 0.5
    w2 = 0.3

    b = -0.2

    val = w1 * x1 + w2 * x2 + b

    if val <= 0:
        return 0
    else:
        return 1



def nand_gate(x1, x2):
    w1 = -0.5
    w2 = -0.3

    b = -0.2

    val = w1 * x1 + w2 * x2 + b

    if val <= 0:
        return 0
    else:
        return 1


x1 = 0.3
x2 = 0.3

print(and_gate(x1, x2))
print(nand_gate(x1, x2))