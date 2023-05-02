import os
import sys

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2, l2
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, \
    Activation, Dropout
from keras.models import Model
from keras.datasets import cifar10

from matplotlib import pyplot as plt

from pathlib import Path
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

img_directory_path = '/media/alpha/Samsung_T5/deepLearning/my_face/'
img_directory_path_object = Path('/media/alpha/Samsung_T5/deepLearning/my_face/')
imgs_path = sorted(img_directory_path_object.glob('*.jpg'))

imgs_link = [str(img) for img in imgs_path]
cnt = len(imgs_link)

# cv2에서 resize할 때는 행렬의 순서를 바꿔야한다.
input_shape = (32, 32)


def img_gen(target_trainset_cnt, train_dataset):
    img_cnt = len(imgs_link)
    for img in imgs_link:
        img_origin = cv2.imread(img)
        img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_origin, dsize=input_shape, interpolation=cv2.INTER_LINEAR)

        yield resized_img
    size = target_trainset_cnt - img_cnt
    for j in range(size):
        yield train_dataset[j]


(ci_train_images, ci_train_labels), (ci_test_images, ci_test_labels) = cifar10.load_data()

train_images = np.array(list(img_gen(1000, ci_train_images)), dtype=np.uint8)
train_labels = np.array([1 if i < 240 else 0 for i in range(1000)], dtype=np.uint8)
train_labels = train_labels.reshape(-1, 1)


# 먼저 데이터들을 pre processing 해준 후에 train, test split 해주면 된다.


def process_one_hot_encoding(train_labels):
    # one hot encoding -> 우리가 ascii 코드를 쓰듯이 각 label을 숫자로 바꿔준다.
    return to_categorical(train_labels)


def get_processed_data(train_origin_images, train_origin_labels):
    # train_images와 train_labels를 알맞게 processing 한다.
    # 이미지는 일반적으로 rgb 255개의 value를 가지고 있는데 이 값을 255로 나누어 float형으로 변환하면 이미지 처리에 적합한 numpy array가 된다.
    processed_labels = train_origin_labels.astype(dtype=np.float32)
    processed_images = train_origin_images / 255.0
    return processed_images, processed_labels


# train 데이터를 다시 검증 데이터로 분할 해준다.
def get_train_test_valid_splitted(train_images, train_labels, valid_size=0.15):
    processed_imgs, processed_labels = get_processed_data(train_images, train_labels)
    oh_labels = process_one_hot_encoding(train_labels)

    tr_images, test_images, tr_labels, test_labels = train_test_split(train_images, oh_labels)
    tr_images, val_images, tr_labels, val_labels = train_test_split(tr_images, tr_labels, test_size=valid_size)

    return tr_images, tr_labels, test_images, test_labels, val_images, val_labels


def create_model(input_shape, verbose=0):
    # feature extract layer

    # 컬러 이미지기 때문에 마지막 3차원 데이터를 3으로 넣어준다. (rgb)
    input_tensor = Input(shape=(input_shape[0], input_shape[0], 3), dtype=np.float32)
    # filters는 feature extract를 하기 위한 filter의 개수를 의미한다.
    # kernel_size는 각 filter의 행렬 사이즈를 의미한다. 일반적으로 (3,3)의 필터가 가장 많이 쓰임
    # 알아두기 depth는 현재 input의 depth가 3이기 때문에 자동으로 depth는 3으로 맞춰질 것이다.
    # filter의 depth도 3이어야 함
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # filter의 depth는 64이다. 128의 각 filter는 depth가 64이다.
    x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 512 filters Conv layer 추가하되 이후 MaxPooling을 적용하지 않고 strides는 2로 변경하여 출력 feature map 크기 조정
    x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # fully connected layer
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=300, activation='relu', kernel_regularizer=l2(1e-5), name='fc1')(x)
    x = Dropout(rate=0.3)(x)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output)

    if verbose:
        model.summary()
    return model


tr_images, tr_labels, test_images, test_labels, val_images, val_labels = get_train_test_valid_splitted(train_images,
                                                                                                       train_labels)

model = create_model(input_shape, 1)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', jit_compile=True, metrics=['accuracy'])

rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min', verbose=1)
# 10번 iteration내에 validation loss가 향상되지 않으면 더 이상 학습하지 않고 종료
ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

history = model.fit(x=tr_images, y=tr_labels, validation_data=(val_images, val_labels), batch_size=32, epochs=30,
                    shuffle=True, callbacks=[rlr_cb, ely_cb])

model.evaluate(test_images, test_labels)


def show_history(history):
    plt.figure(figsize=(8, 4))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 30, 2))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    plt.show()


show_history(history)


def show_images(imgs, labels, cnt, ncols=4):
    fig, axes = plt.subplots(nrows=1, ncols=5)
    for i in range(5):
        axes[i].imshow(train_images[i])

    fig.show()
