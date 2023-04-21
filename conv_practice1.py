from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.datasets import fashion_mnist

import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 학습시킬(train) 데이터와 검증용(test) 데이터를 불러온다.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 각 pixel을 255로 나눠서 0 ~ 1 사이의 값으로 만들어준다.
# 이미지 처리할 때, 효율이 올라간다고 한다.
def pre_process_data(images, labels):
    processed_images = np.array(images / 255.0, dtype=np.float32)
    processed_labels = np.array(labels, dtype=np.float32)
    return processed_images, processed_labels


train_images, train_labels = pre_process_data(train_images, train_labels)
test_images, test_labels = pre_process_data(test_images, test_labels)
# train dataset을 다시 이 dataset을 검증할 수 있는 validation dataset 까지 추가로 split한다.
tr_images, val_images, tr_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.15,
                                                                random_state=2023)

# string인 column을 binary 형태로(즉 컴퓨터가 이해할 수 있는 형태로) encoding한다.
# 즉 ascii 코드를 생각하면 된다. 알파벳은 사실 컴퓨터 입장에서는 숫자를 이진수 형태로 변환한 것을 인식하는 것에 불과하듯이
tr_oh_labels = to_categorical(tr_labels)
val_oh_labels = to_categorical(val_labels)
test_oh_labels = to_categorical(test_labels)

INPUT_SIZE = 28


def create_model():
    # convolution 학습에서는 channel이 항상 마지막이다.
    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))


















