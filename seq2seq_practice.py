import os
import shutil
import time
import zipfile

import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
from keras.utils import to_categorical
import timeit


lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
lines = lines[:60000]

lines.tar = lines.tar.apply(lambda x: '\t' + x + '\n')
src_vocab1 = set(''.join(lines['src'].values))


