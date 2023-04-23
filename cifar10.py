import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from keras import Input

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Activation
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

NAMES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


def show_images(imgs, labels, cnt, ncols=4):
    if cnt // 4:
        nrow = cnt // 4

    fig, ax = plt.subplots(nrows=nrow, ncols=ncols, figsize=(22, 6))
    for i in range(cnt):
        # [STUDY] 2차원을 1차원으로 즉 고차원을 저차원으로 줄여준다.
        label = labels[i].squeeze()
        ax[i // ncols][i % ncols].imshow(train_images[i])
        ax[i // ncols][i % ncols].set_title(NAMES[label])
    fig.show()


def get_proccessed_data(images, labels):
    processed_images = np.array(images / 255.0, dtype=np.float32)
    processed_labels = np.array(labels, dtype=np.float32)
    return processed_images, processed_labels


# train_images, train_labels = get_proccessed_data(train_images, train_labels)
# test_images, test_labels = get_proccessed_data(test_images, test_labels)

# train_labels = train_labels.squeeze()
# test_labels = test_labels.squeeze()

IMAGE_SIZE = 32

input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu')(input_tensor)
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)

x = Conv2D(filters=128, kernel_size=(3,3), padding='same',)







