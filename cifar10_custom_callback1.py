import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.regularizers import l2, l1_l2
import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, BatchNormalization, \
    GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

NAMES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(type(train_images))

exit()
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
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.00001))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 512 filters Conv layer 추가하되 이후 MaxPooling을 적용하지 않고 strides는 2로 변경하여 출력 feature map 크기 조정
    x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # fully connected layer
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=50, activation='relu', kernel_regularizer=l2(0.00001))(x)
    x = Dropout(rate=0.2)(x)
    output = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    if verbose:
        print(model.summary())

    return model


model = create_model(32, True)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# weight를 특정 시점에 저장하는 콜백 함수이다.
mcp_cb = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_weights_only=True, save_best_only=True,
                         mode='min', save_freq=10, verbose=0)

# patience(참을성) - monitor 요소의 향상 혹은 줄어듬(loss일 때는 줄어들어야 하고 accuracy일 때는 향상되어야 함) 변하지 않는 것을 몇번 참을 것인지를 지정하는 argument
# 예를 들어 patience는 3이고, 30에폭에 정확도가 99%였을 때,
# 만약 31번째에 정확도 98%, 32번째에 98.5%, 33번째에 98%라면 모델의 개선이 (patience=3)동안 개선이 없었기에,  ReduceLROnPlateau 콜백함수를 실행합니다.

# min_lr - learning rate의 하한선을 지정하는 argument
# factor = new_lr = old_lr * factor
#
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, verbose=0)

ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

history = model.fit(x=tr_images, y=tr_oh_labels, batch_size=128, epochs=8, shuffle=True, validation_data=(val_images, val_oh_labels),
                    callbacks=[rlr_cb, ely_cb]),

model.evaluate(x=test_images, y=test_labelss)

