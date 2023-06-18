import os
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

# 모든 코드는 console에서 실행 후에 실제 코드로 옮기는 중
news_df = pd.DataFrame({'document': documents})
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")

news_df['clean_doc'] = news_df['clean_doc'].apply(
    lambda x: ' '.join([word.lower() for word in x.split() if len(word) > 3]))

# news_df.dropna(inplace=True)

from keras.preprocessing.sequence import skipgrams
