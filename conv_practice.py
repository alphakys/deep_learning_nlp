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
    # tensorflow는 channel last 방식을 사용한다.
    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 1))
    #                                 None는 batch이다. None은 알아서 계산한다는 의미이면서 동시에 batch이다.
    #                                 28 by 28 filter는 4개이다.
    # feature map은 3차원이다. shape=(  None , 28, 28, 4)
    # [STUDY] filter는 무조건 3차원이다.
    #   channel이 3개라면 각 채널 마다
    #   예를들어 R channel에 (4,4) * (3,3) => result(2,2)
    #   G channel에 (4,4) * (3,3) => result(2,2)
    #   B channel에 (4,4) * (3,3) => result(2,2)
    #   최종적으로 2,2,3의 결과값을 각 element마다 더해서
    #    (2,2,1)의 결과값이 나온다.
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_tensor)
    # pool size = 2 by 2
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu')(x)
    x = MaxPooling2D(2)(x)

    # fully connected layer에 데이터를 넣기 위해서는 1차원으로 만들어줘야한다.
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=150, activation='relu')(x)
    x = Dropout(rate=0.4)(x)
    output = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)

    return model


model = create_model()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=tr_images, y=tr_oh_labels, batch_size=128, validation_data=(val_images, val_oh_labels),
                    # callbacks=[rlp_cb, est_cb],
                    epochs=20)


def show_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    plt.show()


show_history(history)
# feature map -> activation map
