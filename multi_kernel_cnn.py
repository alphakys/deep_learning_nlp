import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

sample_num = 145162

vocab_size = 19416
max_len = 30

embedding_dim = 128
dropout_ratio = (0.5, 0.8)
num_filters = 128
hidden_units = 128

model_input = Input(shape=(max_len,))
x = Embedding(vocab_size, embedding_dim, input_length=max_len, name='embedding_layer1')(model_input)
x = Dropout(dropout_ratio[0])(x)

conv_blocks = []

for sz in [3, 4, 5]:
    conv = Conv1D(filters=num_filters,
                  kernel_size=sz,
                  padding='valid',
                  activation='relu',
                  strides=1)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

# [STUDY] A `Concatenate` layer requires inputs with matching shapes except for the concatenation axis.
#   즉 합치려고 하는 차원을 제외하고 shape이 같아야 한다고 한다.
x = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
x = Dropout(dropout_ratio[1])(x)
x = Dense(hidden_units, activation='relu')(x)

model_output = Dense(1, activation='sigmoid')(x)

model = Model(model_input, model_output)
model.compile(metrics=['acc'], optimizer='adam', loss='binary_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('CNN_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=2, callbacks=[es, mc])