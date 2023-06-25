import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

df = pd.read_csv('~/문서/dart.csv')
for k, v in enumerate(df.iterrows()):
    if k < 10:
        TaggedDocument
        print(k, v[1][['name', 'business']])
