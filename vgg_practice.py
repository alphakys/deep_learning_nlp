import os

import cv2
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Dense, GlobalAveragePooling2D
from keras import Input, Model
from keras.applications.vgg16 import VGG16

import numpy as np

from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

IMAGE_SIZE = 32
# pretrained 모델을 사용에 적합한 size로 resize한다.
DESTINATION_SIZE = 64

# train / test image를 로드하고 image와 label을 preprocessing 한다.

def get_oh_encoding(train_labels, test_labels):
    return to_categorical(train_labels), to_categorical(test_labels)

def preprocessing_data(train_images, test_images, train_labels, test_labels):
    # image는 255.0으로 나누어서 0~1사이의 값으로 processing 하고
    # label은 one_hot_encoding 한다.

    cv2.resize(dsize=(64, 64))
    train_processed_images = np.array(train_images/255.0, dtype=np.flot32)
    test_processed_images = np.array(test_images / 255.0, dtype=np.flot32)



    return tr_images, test_images,


print(train_images.shape, test_images.shape)


#
#
# IMAGE_SIZE = 32
# BATCH_SIZE = 64
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

