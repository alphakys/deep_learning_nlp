import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


# [STUDY] filter는 채널별로 순회하며 feature map을 만든다.
#   만약에 RGB의 세개의 채널이 있다면
#   R -> (3,3) // G -> (3,3) // B -> (3,3)이 만들어졌다면
#   각각의 행과 열에 맞춰 다시 곱을 하면 최종 feature map이 만들어진다.
#   activation map은 feature map에 activation function(like relu)를 적용한 결과이다.
#   따라서 최종결과는 activation map이 나온다.
#   pooling은 activation map에 대해서 적용한다.
#   pooling을 적용함으로 인해 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용한다.


def create_model():
    # [STUDY] convolution 학습을 시키기 위해서는 3차원의 input shape을 설정해줘야 한다.
    #   그리고 channel이 항상 마지막이다.
    #   batch까지 4차원을 받는데 input에서는 batch를 무시한다.
    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    # 결국 filter가 weight의 행렬임을 알자
    # [STUDY] !!
    #   결국 CONVOLUTION 연산은 INPUT과 FILTER의 DEPTH가 같아야 한다.!!!
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_tensor)
    # x = Dropout(rate=0.5)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    # [STUDY] 입력 데이터의 행 크기와 열 크기는 Pooling 사이즈의 배수(나누어 떨어지는 수)여야 합니다
    #   Pooling 크기가 (2, 2) 라면 출력 데이터 크기는 입력 데이터의 행과 열 크기를 2로 나눈 몫입니다.
    #   pooling 크기가 (3, 3)이라면 입력데이터의 행과 크기를 3으로 나눈 몫이 됩니다.
    x = MaxPooling2D(input_shape=(2, 2))(x)

    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=100, activation='relu')(x)
    output = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)

    return model

def show_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    plt.show()

model = create_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=tr_images, y=tr_oh_labels, validation_data=(val_images, val_oh_labels), batch_size=500, epochs=3)

show_history(history)
