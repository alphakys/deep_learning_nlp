import os

from keras.engine.keras_tensor import KerasTensor
from keras.optimizers import Adam

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import Input, Model
from keras.layers import Flatten, Dense

### 앞에서 생성한 로직들을 함수화
# * Functional API로 모델 만들기
# * pixel값 1 ~ 255를 0 ~ 1사이값 Float 32로 만들기
# * One Hot Encoding Label에 적용하기
# * 학습과 검증 데이터로 나누기.
# * compile, 학습/예측/평가

INPUT_SIZE = 28


# [STUDY] 핵심 Once the model is created, you can config the model
#   with losses and metrics with model.compile(), train the model with model.fit(),
#   or use the model to do prediction with model.predict().
def create_model(input_size):
    # is used to instantiate a Keras tensor
    # input은 차원이 아니라 처리하고자 하는 데이터의 행렬(즉 shape)을 설정한다.
    input_tensor = Input(shape=(input_size, input_size), name='input_tensor1')
    # flatten을 통해서 28 by 28의 각 pixel을 784개의 feature로 나열한다.
    x: KerasTensor = Flatten()(input_tensor)
    # 첫번째 dense에 node가 100개 activation 함수는 relu
    x = Dense(units=100, activation='relu')(x)
    # 두번째 dense에 node가 30개 activation 함수는 relu
    x = Dense(units=30, activation='relu')(x)
    # 출력은 10개의 패션 아이템에 대한 카테고리를 분류해야 하니 softmax(다중 classfication)
    output = Dense(units=10, activation='softmax')(x)
    mo = Model(inputs=input_tensor, outputs=output)

    return mo


# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model = create_model(INPUT_SIZE)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# callback function
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

mcp_callback = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss: 2f}.hdf5', monitor='val_loss', mode='min',
                               save_best_only=True, verbose=1, period=5)

rrl_callback = ReduceLROnPlateau(factor=0.2, monitor='val_loss', patience=5, mode='min')

est_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

history = model.fit(callbacks=[est_callback, rrl_callback, mcp_callback], x=..., y=..., validation_data=..., epochs=20, batch_size=32)


#
#
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2023-04-18 12:25:35.017665: I tensorflow/core/platform/cpu_feature_guard.cc:193]
# This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to
# use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
#
# 2023-04-18 12:25:35.018654: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting:
#
#     2. Tune using inter_op_parallelism_threads for best performance.










