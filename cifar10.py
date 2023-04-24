import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from keras import Input

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
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


train_images, train_labels = get_proccessed_data(train_images, train_labels)
test_images, test_labels = get_proccessed_data(test_images, test_labels)

# label 데이터가 2차원임. 이를 Keras 모델에 입력해도 별 문제없이 동작하지만, label의 경우는 OHE적용이 안되었는지를 알 수 있게 명확하게 1차원으로 표현해 주는것이 좋음.
# 2차원인 labels 데이터를 1차원으로 변경.
train_labels = train_labels.squeeze()
test_labels = test_labels.squeeze()

IMAGE_SIZE = 32

# 3차원으로 input 입력함 channel_depth는 3이다.
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
# input_tensor가 depth가 3이기 때문에 자연스럽게 filter의 depth도 3이다.
# shape = (5, 5, 3) * 32개의 필터가 있다.
# padding은 valid이기 때문에 input 행렬의 크기는 줄어든다.
# 32 - 5 +1 = 28
# activation 함수도 설정해줬기 때문에 feature map => activation map
x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
# shape = (None, 28, 28, 32)의 input이 들어오고 여기에 filter shape = (3, 3, 32) * 32개의 필터가 생성됨
# 왜냐하면 input의 depth가 32이기 때문에

x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
# pooling 과정을 통해서 원하는 부분을 강조하거나 큰 데이터 행렬 사이즈를 줄일 수 있다.
# 결과는 행 /2 열 / 2가 된다.
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# 3차원의 input을 1차원으로 길게 늘여준다.
x = Flatten(name='flatten')(x)
# input node의 비율을 0.5만큼 줄인다. 지나친 overfitting을 막기 위해서
x = Dropout(rate=0.5)(x)
x = Dense(units=300, activation='relu', name='fc1')(x)
x = Dropout(rate=0.3)(x)
output = Dense(units=10, activation='softmax', name='output')(x)

model = Model(inputs=input_tensor, outputs=output)

model.summary()

# [STUDY] label값이 원-핫 인코딩 되지 않았기 때문에 model.compile()에서 loss는 반드시 sparse_categorical_crossentropy여야함.
#   만일 label값이 원-핫 인코딩 되었다면 loss는 categorical_crossentropy 임.
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_images, y=train_labels, batch_size=300, epochs=5, validation_split=0.15)


def show_history(history):
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0,1,0.05))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')

    plt.legend()
    plt.show()

show_history(history)

model.evaluate(test_images, test_labels)

# [STUDY] predict할 때도 input data는 4차원으로 들어가야함
#   4차원 데이터를 학습했기 때문에
#   np.expand_dims ==> axis는 늘리고 싶은 차원의 index를 넣어주면 됨
#   예를들면 axis=0이면 현재 3차원이기 때문에 (32, 32, 3) ==>>>> (1, 32, 32, 3) 결과가 나옴
#   그리고 결과값은 10개의 클래스를 가지는 softmax이기 때문에 각 클래스에 대한 확률이 나온다.
#   그 중에서 max번째의 index가 클래스에 해당하겠지?

# 한 개를 예측할 때는 아래와 같이 4차원으로 설정해서 예측해야함
prd = model.predict(np.expand_dims(test_images[0], axis=0))
# 그러나 batch_size를 사용한다면 4차원으로 알아서 인식하고 만들어주는 듯하다.
prd = model.predict(test_images[:32], batch_size=32)

prd_class = np.argmax(prd, axis=1)
# !! 해당 클래스가 나옴 !!
print(NAMES[prd.argmax()])

# [STUDY] scale=표준편차, loc=평균,
np.random.normal(loc=0.0, scale=0.01, size=(100,100))