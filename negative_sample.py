import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.preprocessing.sequence import skipgrams

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

# 아무 내용없는 series에 대해서 NaN 처리하기
news_df.replace("", float("NaN"), inplace=True)
# NaN drop하기
news_df.dropna(inplace=True)

stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()
drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)

tk = Tokenizer()
tk.fit_on_texts(tokenized_doc)

word_to_index = tk.word_index
index_to_word = tk.index_word

encoded = tk.texts_to_sequences(tokenized_doc)
vocab_size = len(tk.word_index) + 1

skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]
