import numpy as np
from keras import Input

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

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


# label에 one hot encoding 적용 및 이미지 리스트에는 각 rgb 채널 값에 대하여 255로 나눠서
# 데이터 학습에 유리하게 만듬 // label은 float으로 바꿔준다.
def get_preprocessed_ohe(images, labels):
    processed_imgs, processed_label = get_proccessed_data(images, labels)
    oh_labels = to_categorical(processed_label)

    return processed_imgs, oh_labels


# 학습 / 검증 / 테스트 데이터로 분할해준다.
def get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_test_size=0.15,
                             random_seed=2023):
    train_imas, train_labelss = get_preprocessed_ohe(train_images, train_labels)
    test_imags, test_labelss = get_preprocessed_ohe(test_images, test_labels)
    tr_images, val_images, tr_oh_labels, val_oh_labels = train_test_split(train_imas, train_labelss,
                                                                          test_size=valid_test_size,
                                                                          random_state=random_seed)

    return (tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_imags, test_labelss)


(tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_imags, test_labelss) = get_train_valid_test_set(
    train_images, train_labels, test_images, test_labels)


def create_model(input_shape_size, verbose=False):
    input_tensor = Input(shape=(input_shape_size, input_shape_size, 3))
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # fully connected layer

    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=300, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    output = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    if verbose:
        print(model.summary())

    return model


model = create_model(32, True)



# label 데이터가 2차원임. 이를 Keras 모델에 입력해도 별 문제없이 동작하지만, label의 경우는 OHE적용이 안되었는지를 알 수 있게 명확하게 1차원으로 표현해 주는것이 좋음.
# 2차원인 labels 데이터를 1차원으로 변경.
train_labels = train_labels.squeeze()
test_labels = test_labels.squeeze()

IMAGE_SIZE = 32
