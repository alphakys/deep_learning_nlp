import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import urllib.request
from matplotlib import pyplot as plt
from collections import Counter
from konlpy.tag import Okt, Mecab
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename='steam.txt')

# read_table can read tab separated data and names -> coumns name
total_data = pd.read_table('steam.txt', names=['label', 'reviews'])

total_data.drop_duplicates(subset=['reviews'], inplace=True)  # reviews 열에서 중복인 내용이 있다면 중복 제거

train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
train_data['label'].value_counts().plot(kind='bar')

# 한글과 공백을 제외하고 모두 제거
train_data['reviews'] = train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['reviews'].replace('', np.nan, inplace=True)

test_data.drop_duplicates(subset=['reviews'], inplace=True)  # 중복 제거
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거

stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지',
             '임', '게', '만', '게임', '겜', '되', '음', '면']
