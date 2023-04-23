import os

from keras import Input
from keras.layers import Conv2D, ZeroPadding2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_tensor = Input(shape=(5, 5, 1))
x = Conv2D(filters=1, kernel_size=3, strides=1)(input_tensor)

# [STUDY] OUTPUT 구하기
#   O = (I - F + 2P)/2 + 1 = (5 - 3 + 0 )/1 + 1 = 3


input_tensor = Input(shape=(6, 6, 1))
# [STUDY] 만약에 zeropadding2d에 튜플 안에 두 개의 튜플이 있다면 그것은 (up, down)padding // (left, right)padding이다.
padded_output = ZeroPadding2D(padding=((1,1),(1,1)))
x = Conv2D(filters=1, kernel_size=3, strides=2, padding='same')(input_tensor)