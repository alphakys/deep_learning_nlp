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

skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers import Dot
from keras.utils import plot_model
from IPython.display import SVG

embedding_dim = 100

# 중심단어를 위한 임베딩 테이블 -> 중심단어(label이 1인 단어 pair)
w_inputs = Input(shape=(1,), dtype='int32')
# call을 통해 Input을 통해 설정한 config를 주입한다.
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

# 주변단어를 위한 임베딩 테이블 -> 주변단어(label이 0인 단어 pair)
c_inputs = Input(shape=(1,), dtype='int32')
context_embedding = Embedding(vocab_size, embedding_dim)(c_inputs)

dot_product = Dot()










# from collections import Counter

# gram1_to_word = [[index_to_word[w] for w in l] for l in skip_grams[0][0]]
#
# flatten_list = []
# [flatten_list.extend(word_l) for word_l in gram1_to_word]
#
# counter_token_doc = Counter(tokenized_doc[0])
# counter_gram1 = Counter(flatten_list)
#
# for k, v in counter_token_doc.items():
#     counter_token_doc[k] = (v, counter_gram1[k])
