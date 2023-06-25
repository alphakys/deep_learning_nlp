import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

from gensim.models import Word2Vec
from tqdm import tqdm
from konlpy.tag import Mecab


wiki_file_path = []
for direc in os.listdir('/home/alpha/문서/text/'):
    base_path = '/home/alpha/문서/text/'
    wiki_upper_path = base_path + direc

    for wikifile in os.listdir(wiki_upper_path):
        wiki_file_path.append(wiki_upper_path + '/' + wikifile)

with open('out_file.txt', 'w') as outfile:
    for file in wiki_file_path:
        f = open(file, 'r')
        contents = f.read()
        outfile.write(contents)


mecab = Mecab()

f = open('out_file.txt')

lines = f.read().splitlines()

# 메모리를 너무 잡아먹어서 generator 패턴으로 돌렸지만 그래도 메모리 100%로 차고
# swap 영역으로 넘어가서 학습 다 시키지 못하고 도중에 종료함
result = list((mecab.morphs(line) for line in tqdm(lines) if line))