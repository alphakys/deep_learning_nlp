import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

df = pd.read_csv('dart.csv')

mecab = Mecab()
tagged_corpus_list = []

# cleaning dataframe
df = df[[not _ for _ in df['business'].isnull()]]
df.reset_index(drop=True, inplace=True)

for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['business']
    tag = row['name']
    tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))


from gensim.models import doc2vec
model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=-1, window=8)
model.build_vocab(tagged_corpus_list)

model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=20)

