import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from numpy import ndarray

from keras.datasets import fashion_mnist
from keras.layers import Flatten, Dense, Layer
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import Input, Model

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

INPUT_SIZE = 28


def create_model(input_size):
    # [STUDY] INPUT의 shape을 만들 때, 처리하고자 하는 데이터 한 행렬을 기준으로 만든다.
    #   차원이 아니라
    #   즉 fasion_mnist 모델을 예로든다면 60000개의 차원(6만개의 데이터세트)이 있고 각각이 28 by 28의 행렬이다.
    #   여기서 28 by 28의 행렬을 shape에 넣어줘야한다.

    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE))
    # flatten 해줄 때, input 값으로 call argument에 넣어준다.
    # flatten을 하면 1
    x = Flatten()(input_tensor)
    x = Dense(units=100, activation='relu')(x)
    x = Dense(units=30, activation='relu')(x)
    output = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output, name='alpha_practice', )
    return model


model = create_model(INPUT_SIZE)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# [STUDY] ModelCheckpoint(filepath, monitor=‘val_loss’, verbose=0, save_best_only=False, save_weights_only=False, mode=‘auto’, period=1)
#   특정 조건에 맞춰서 모델을 파일로 저장
#   filepath: filepath는 (on_epoch_end에서 전달되는) epoch의 값과 logs의 키로 채워진 이름 형식 옵션을 가질 수 있음. 예를 들어 filepath가 weights.{epoch:02d}-{val_loss:.2f}.hdf5라면,
#   파일 이름에 세대 번호와 검증 손실을 넣어 모델의 체크포인트가 저장
#   monitor: 모니터할 지표(loss 또는 평가 지표)
#   save_best_only: 가장 좋은 성능을 나타내는 모델만 저장할 여부
#   save_weights_only: Weights만 저장할 지 여부
#   mode: {auto, min, max} 중 하나. monitor 지표가 감소해야 좋을 경우 min, 증가해야 좋을 경우 max, auto는 monitor 이름에서 자동으로 유추.
#   model.fit(x=tr_images, y=tr_oh_labels, epochs=20, batch_size=32, verbose=1)

mcp_cb = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss: 2f}.hdf5', monitor='val_loss', save_best_only=True,
                         mode='min',
                         period=5, verbose=1, save_weights_only=True, model='model.h5')

# [STUDY] ReduceLROnPlateau(monitor=‘val_loss’, factor=0.1, patience=10, verbose=0, mode=‘auto’, min_delta=0.0001, cooldown=0, min_lr=0)
#   특정 epochs 횟수동안 성능이 개선 되지 않을 시 Learning rate를 동적으로 감소 시킴
#   monitor: 모니터할 지표(loss 또는 평가 지표)
#   factor: 학습 속도를 줄일 인수. new_lr = lr * factor
#   patience: Learing Rate를 줄이기 전에 monitor할 epochs 횟수.
#   mode: {auto, min, max} 중 하나. monitor 지표가 감소해야 좋을 경우 min, 증가해야 좋을 경우 max, auto는 monitor 이름에서 유추.

rlp_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='min', verbose=1)

# [STUDY] EarlyStopping(monitor=‘val_loss’, min_delta=0, patience=0, verbose=0, mode=‘auto’, baseline=None, restore_best_weights=False)
#   특정 epochs 동안 성능이 개선되지 않을 시 학습을 조기에 중단
#   monitor: 모니터할 지표(loss 또는 평가 지표)
#   patience: Early Stopping 적용 전에 monitor할 epochs 횟수.
#   mode: {auto, min, max} 중 하나. monitor 지표가 감소해야 좋을 경우 min, 증가해야 좋을 경우 max, auto는 monitor 이름에서 유추.

est_cb = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)
history = model.fit(x=tr_images, y=tr_oh_labels, batch_size=50, validation_data=(val_images, val_oh_labels),
                    callbacks=[mcp_cb, rlp_cb, est_cb],
                    epochs=40)



















#
# # functoinal api practice
# INPUT_SIZE = 28
#
# model = Sequential([
#     Flatten(input_shape=(INPUT_SIZE, INPUT_SIZE)),
#     Dense(100, activation='relu'),
#     Dense(30, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#
# model.summary()
#
# model1 = Sequential()
# model1.add(Flatten(input_shape=(INPUT_SIZE, INPUT_SIZE)))
# model1.add(Dense(100, activation='relu'))
# model1.add(Dense(30, activation='relu'))
# model1.add(Dense(10, activation='softmax'))
#
#
# model1.summary()
#
# print('🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉🐲🐉')
#
# # functional api와 sequential 사이의 차이점은 input layer를 설정한다는 점이다.
# input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE))
# # flatten에 생성인자는 init argument에 입력값은 call argument에 넣어준다.
# x = Flatten()(input_tensor)
# x = Dense(100, activation='relu', name='Dense1')(x)
# x = Dense(30, activation='relu')(x)
# output = Dense(10, activation='softmax')(x)
#
# model2 = Model(input_tensor, output)
#
# model2.summary()
#
# import tensorflow as tf
#
# class CustomDense(Layer):
#     # 생성자에서 class 생성에 핵심적인 argument를 받는다.
#     def __init__(self, units=32):
#         # unit default는 32개
#         super(CustomDense, self).__init__()
#         self.units = units
#
#     def build(self, input_shape):
#         self.w = self.add_weight(
#             name='custom_practice',
#             shape=(input_shape[-1], self.units),
#             initializer='random_normal',
#             trainable=True
#         )
#         self.b = self.add_weight(
#             shape=(self.units,), initializer='random_normal', trainable=True
#         )
#
#     # innstance가 생성된 후, call function에서 input argument를 받고 output을 return한다.
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
