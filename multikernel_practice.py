import urllib
import warnings

from keras.utils import pad_sequences
from keras_preprocessing.text import Tokenizer
from konlpy.tag import Okt
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

vocab_size = 19416
max_len = 30

embedding_dim = 128
dropout_ratio = [0.5, 0.8]
num_filters = 128
hidden_units = 128

model_input = Input()
