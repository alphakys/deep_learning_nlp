import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy

from matplotlib import pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def show_images(images, labels, ncols=8):
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(22, 6))
    for i in range(ncols):
        label_name = class_names[labels[i]]
        print(f'{i} {label_name}')
        ax[i].imshow(images[i, :, :], cmap='gray')
        ax[i].set_title(label_name)


def get_preprocessed_data(images, labels):
    # array이의 즉 한 pixel의 색상값의 최대인 255로 나누면 0 ~ 1 사이의 값이 나온다.
    # 그러면 왜 0 ~ 1 사이의 값으로 전처리 해주는걸까?
    # [STUDY] np.array에서 object-like라고 매개변수를 넣을 수 있다고 되어있는데
    #   numpy가 리스트 전체를 한 번에 처리할 수 있는 듯하다.
    processed_img = np.array(object=(images / 255.0), dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return processed_img, labels

train_images, train_labels = get_preprocessed_data(train_images, train_labels)

IMPUT_SIZE = 28
model = Sequential([
    # [STUDY] flatten은 말 그대로(평탄화) 28 * 28의 784개의 픽셀을 1차원으로 만든다.
    #   (None, 784)
    #   그리고 이 784개의 픽셀을 feature로 사용한다고 한다.
    Flatten(input_shape=(IMPUT_SIZE, IMPUT_SIZE)),
    # Dense는 neural network에 사용되는 layer이다.
    # unit은 node의 개수를 의미
    # activation은 활성화 함수를 의미하며 activation이 설정되어 있지 않으면
    # weighted sum까지만 이루어지고 activation도 설정되어 있으면 activation까지 이루어진다.
    Dense(units=100, activation='relu'),
    Dense(units=30, activation='relu'),
    Dense(units=10, activation='softmax')
])

# adam optimizer를 이용하고 alpha(learning_rate)는 0.001로 설정한다.
# loss(손실함수, 목적함수, 비용함수)는 CategoricalCrossentropy를 사용한다.
# metrics는 성능측정을 의미한다. 여기서는 accuracy를 사용한다.
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

from keras.utils import to_categorical

# [STUDY] one-hot encoding은 column이 string으로 되어있는 것을 int로 구분할 수 있게 만드는 것이다.
#   예를 들어 ['sun', 'moon', 'umbrella']가 있을 때, sun은 0, moon은 1, umbrella는 2로 인코딩한다.
#   우리가 마치 ascii 코드표에서 a가 97이 듯이 마찬가지로 각 label을 숫자로 인코딩하는 것으로 이해된다.
train_oh_labels = to_categorical(train_labels)
test_oh_labels = to_categorical(test_labels)

history = model.fit(x=train_images, y=train_oh_labels, epochs=20, batch_size=32, verbose=1)

# predict할 때, 애초에 fit을 시킬 때, 3차원 데이터를 넣어서 fit 했기 때문에 predict하는 데이터도 3차원으로 매개변수에 넣어준다.
# 이 때, 사용하는 함수가 expand_dims이다. axis는 0이면 차원, 1이면 행, 2이면 열이다.
prd_proba = model.predict(np.expand_dims(test_images[0], axis=0))

print("softmax output : ", prd_proba)

# argmax는 가장 큰 값을 가진 index를 반환한다.
pred = np.argmax(np.squeeze(prd_proba))

# !! 테스트 데이터 세트로 모델 성능 검증
model.evaluate(test_images, test_oh_labels, batch_size=64)





