import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/data.csv", filename="data.csv")

df = pd.read_csv('data.csv')


def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


def make_lower_case(text):
    return text.lower()


def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


df['cleaned'] = df['Desc'].apply(_removeNonAscii)
df['cleaned'] = df.cleaned.apply(make_lower_case)
df['cleaned'] = df.cleaned.apply(remove_stop_words)
df['cleaned'] = df.cleaned.apply(remove_punctuation)
df['cleaned'] = df.cleaned.apply(remove_html)

df['cleaned'].replace('', np.nan, inplace=True)
df = df[df['cleaned'].notna()]

corpus = []
for words in df['cleaned']:
    corpus.append(words.split())

# GoogleNewsVectors file은 300차원의 vector이다.
word2vec_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
word2vec_model.build_vocab(corpus)
# [STUDY] https://stackoverflow.com/questions/67707418/index-out-of-bounds-error-with-gensim-4-0-1-word2vec-model
#   lockf 설정 안되는 문제 해결 stackoverflow
word2vec_model.wv.vectors_lockf = np.ones(len(word2vec_model.wv), dtype=np.float32)
word2vec_model.wv.intersect_word2vec_format('/home/alpha/문서/GoogleNews-vectors-negative300.bin.gz', lockf=1.0,
                                            binary=True)


def get_document_vectors(document_list):
    document_embedding_list = []

    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word2vec_model.wv.key_to_index.get(word, None):
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model.wv.get_vector(word)
                else:
                    doc2vec = doc2vec + word2vec_model.wv.get_vector(word)
        if doc2vec is not None:
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
    return document_embedding_list


document_embedding_list = get_document_vectors(df['cleaned'])
cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)

def recommendations(title):
    library = {v: k for k, v in df['title'].iteritems()}
    bookof_idx = library[title]

    book_sim = {v: df['title'][idx] for idx, v in enumerate(cosine_similarities[bookof_idx]) if idx != 2017}
    # [STUDY] dict를 정렬하기 위해선 items()를 사용해야 한다.
    book_sim = sorted(book_sim.items(), reverse=True)

    return book_sim[:5][1]


