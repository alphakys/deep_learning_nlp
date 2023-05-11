import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.applications import ResNet50V2, Xception
from numpy import ndarray
from sklearn.model_selection import train_test_split

import cv2
from keras.utils import to_categorical

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

import numpy as np

from keras.datasets import cifar10


def get_preprocessed_data(images, labels, scaling=True):
    if scaling:
        images = np.array(images, dtype=np.float32) / 255.0
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return images, labels


def get_preprocessed_ohe(images, labels):
    images, labels = get_preprocessed_data(images, labels, scaling=False)
    oh_labels = to_categorical(labels)
    return images, oh_labels


def get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.15):
    train_images, train_oh_labels = get_preprocessed_ohe(train_images, train_labels)
    test_images, test_oh_labels = get_preprocessed_ohe(test_images, test_labels)

    tr_images, val_images, tr_oh_labels, val_oh_labels = train_test_split(train_images, train_oh_labels,
                                                                          test_size=valid_size)

    return tr_images, tr_oh_labels, val_images, val_oh_labels, test_images, test_oh_labels


# 입력 image의 크기를 resize 값 만큼 증가. CIFAR10의 이미지가 32x32로 작아서 마지막 feature map의 크기가 1로 되어 모델 성능이 좋지 않음.
# 마지막 feature map의 크기를 2로 만들기 위해 resize를 64로 하여 입력 이미지 크기를 변경. 단 메모리를 크게 소비하므로 64이상은 kernel이 다운됨.
def get_resized_images(images, resize=64):

    resized_images = list(cv2.resize(img, (resize, resize)) for img in images)
    resized_images = np.array(resized_images)
    return resized_images


def create_base_model(input_size, model_name='vgg16', verbose=False):
    input_tensor = Input(shape=(input_size, input_size, 3))
    print(input_tensor)

    if model_name == 'vgg16':
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_name == 'resnet50':
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif model_name == 'xception':
        base_model = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)

    if verbose:
        print(base_model.summary())

    bm_output = base_model.output

    x = GlobalAveragePooling2D()(bm_output)
    x = Dense(50, activation='relu')
    if model_name != 'vgg16':
        x = Dropout(0.5)(x)
    x = Dropout(0.5)(x)

IMAGE_SIZE = 64
BATCH_SIZE = 64

def do_cifar10_train_evaluation(image_size=IMAGE_SIZE, model_name='vgg16'):

    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    tr_images, tr_oh_labels, val_images, val_oh_labels, test_images, test_oh_labels = \
        get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_size=0.15)

    if image_size > 32:
        tr_images = get_resized_images(tr_images)
        val_images = get_resized_images(val_images)
        test_images = get_resized_images(test_images)

    create_base_model(image_size, model_name)


#
# input_tensor = (IMAGE_SIZE, IMAGE_SIZE, 3)
# # cifar는 classification이 10개이기 때문에 imagenet의 full conncected layer를 사용할 필요가 없다.
# # 중요! 따라서 include_top을 false로 설정한다.
# # 결국 라이브러리를 사용하는 것은 기존에 잘 만들어진 모델을 이용하면 처음부터 모델을 만들지 않아도 된다는 장점이 있다.
# # 그리고 효율이 좋다. 내가 원하는 부분만 수정해서 사용할 수 있다는 점이다.
# # 커스텀 모델에는 include_top을 모두 False로 설정할 것이다.
# base_model = VGG16(input_shape=input_tensor, include_top=False, weights='imagenet')
# bm_output = base_model.output
#
# # base_model의 output을 input으로 classification layer를 재구성.
# # global average pooling은 input 값의 depth에 해당하는 각 feature map의 평균을 구해서
# # 따라서 2차원의 행렬이 1차원의 벡터로 변환된다.
# x = GlobalAveragePooling2D()(bm_output)
#
# x = Dense(50, activation='relu')(x)
#
# output = Dense(10, activation='softmax')(x)
#
# # base_model.input을 호출하면 input_1 (InputLayer) [(None, 32, 32, 3)]
# # 처음 input layer가 반환됨
# model = Model(inputs=base_model.input, outputs=output)
#
# model.summary()


# [STUDY]
#   NUMPY MEAN
#    t = np.array([[[1,1,1], [2,2,2], [3,3,3]],[[4,4,4], [5,5,5], [6,6,6]]])
#    array([[[1, 1, 1],
#            [2, 2, 2],
#            [3, 3, 3]],
#           [[4, 4, 4],
#            [5, 5, 5],
#            [6, 6, 6]]])
#    t.shape
#    Out[49]: (2, 3, 3)
#    np.mean(t)
#    Out[50]: 3.5
#    np.mean(t, axis=0)
#    axis 0 -> 각 row별로 평균을 낸다.
#    axis 1 -> 각 column별로 평균을 낸다. => 각 3차원 마다 (1,2,3), (1,2,3), (1,2,3)
#                                                     (4,5,6), (4,5,6), (4,5,6)
#    3차원 matrix의 각 차원 마다의 평균값 => np.mean(t, axis=(2,1))
