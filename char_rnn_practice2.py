import os
import warnings
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import tensorflow as tf

from string import punctuation

from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

from keras.utils import pad_sequences, to_categorical
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, TextVectorization
from keras.models import Sequential

raw_text = '''
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
'''

tokens = raw_text.split()
sentences = ' '.join(tokens)

index_to_char = {idx: v for idx, v in enumerate(sorted(list(set(sentences))))}
char_to_index = {v: idx for idx, v in enumerate(sorted(list(set(sentences))))}
