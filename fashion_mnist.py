import numpy as np

from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

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
    processed_img = np.array(object=images / 255.0, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    return processed_img, labels


train_images, train_labels = get_preprocessed_data(train_images, train_labels)
test_images, test_labels = get_preprocessed_data(test_images, test_labels)
# [STUDY] train_test_split(feature_dataset, target_dataset, random_state는 random seed를 의미한다.)
tr_images, val_images, tr_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.15,
                                                                random_state=2021)

# to_categorical은 one-hot encoding을 해주는 함수이다.
# 결국 우리가 ascii 코드는 알파벳읇 비롯하여 특수문자들을 컴퓨터가 인식할 수 있는 숫자 코드로 변환해주는 코드표이듯이 e.g: a -> 97 b -> 98
# label(string)을 숫자 코드표로 컴퓨터가 인식할 수 있는 형태로 바꿔주는 것이다.
tr_oh_labels = to_categorical(tr_labels)
val_oh_labels = to_categorical(val_labels)

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


# 텐서플로우 소스코드 빌드 작업
# 회사 컴퓨터에서 했던 작업 반복

model.summary()


# validation data는 tuple로 넣어준다.
history = model.fit(x=tr_images, y=tr_oh_labels, epochs=20, batch_size=128, validation_data=(val_images, val_oh_labels),
                    verbose=1)

print(history.history['loss'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['val_accuracy'])

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape the data
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)
# Convert the labels to categorical data
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# Define the model
model = Sequential()
model.add(SimpleRNN(units=128, input_shape=(28, 28)))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, epochs=10, verbose=0)
# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Get the states for each step
states = []
for i in range(len(x_test)):
    states.append(model.predict_state_at(x_test[i]))
# Print the states
for state in states:
    print(state)
#
# # predict할 때, 애초에 fit을 시킬 때, 3차원 데이터를 넣어서 fit 했기 때문에 predict하는 데이터도 3차원으로 매개변수에 넣어준다.
# # 이 때, 사용하는 함수가 expand_dims이다. axis는 0이면 차원, 1이면 행, 2이면 열이다.
# prd_proba = model.predict(np.expand_dims(test_images[0], axis=0))
#
# print("softmax output : ", prd_proba)
#
# # argmax는 가장 큰 값을 가진 index를 반환한다.
# pred = np.argmax(np.squeeze(prd_proba))
#
# # !! 테스트 데이터 세트로 모델 성능 검증
# model.evaluate(test_images, test_oh_labels, batch_size=64)
