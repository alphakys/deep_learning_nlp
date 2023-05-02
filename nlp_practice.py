import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.text import text_to_word_sequence

import pandas as pd
from ydata_profiling import ProfileReport

from nltk.tokenize import word_tokenize

# [STUDY] pandas profiling 예제
# spam_data = pd.read_csv('spam.csv', encoding='latin-1')
# pf = ProfileReport(df=spam_data)
#
# pf.to_file('spam_data.html')

text = "Jake's Time is an illusion. Lunchtime doulbe so!"
s = word_tokenize(text=text)
s1 = WordPunctTokenizer().tokenize(text)

text_to_word_sequence(text)
["jake's", 'time', 'is', 'an', 'illusion', 'lunchtime', 'doulbe', 'so']
text_to_word_sequence("we're family")
["we're", 'family']
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize("we're family")

