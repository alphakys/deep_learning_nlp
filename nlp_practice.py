import os

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
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

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has',
         'starting']

# print('표제어 추출 전 :', words)
# print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in words])

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person." \
           " he Knew A Secret! The Secret He Kept is huge secret. Huge secret. " \
           "His barber kept his word. a barber kept his word. His barber kept his secret." \
           " But keeping and keeping such a huge secret to himself was driving the barber crazy. " \
           "the barber went up a huge mountain."

sentences = sent_tokenize(raw_text)

stop_words = set(stopwords.words('english'))

vocab = {}
preprocessed_sentences = []

for sen in sentences:
    # convert all the words to lower case
    sen = sen.lower()
    # tokenize the lower caseed sentence
    tokenized_sentence = word_tokenize(sen)

    # append the tokenized word that filtering stop words and smaller than 2 digits
    result = []

    for word in tokenized_sentence:
        # filtering stop words
        if word not in stop_words:
            # filtering words that smaller than 2 digits
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)
print(preprocessed_sentences)

# [STUDY] how to sort dictionary by value
#   when I should copy list data, must use list slicing => new_list = old_list[:]
frequency_list = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

for enum in enumerate(frequency_list):
    print(f'importance : {enum[0]+1}, tuple : {enum[1]}')


word_to_index = [enum for enum in enumerate(frequency_list) if enum[0]+1 <= 5]
word_to_index['OOV'] = len(word_to_index) + 1

from collections import Counter
all_words_list = sum(preprocessed_sentences, [])

# counting all words by frequency
new_vocab = Counter(all_words_list)
# filtering top 5 words
new_vocab = new_vocab.most_common(5)

from nltk import FreqDist
import numpy as np

# using numpy function hstack to make 2d array 1d array => flatten
vocab = np.hstack(preprocessed_sentences)
vocab = FreqDist(vocab)

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(preprocessed_sentences)


vocab_size = 5
# if I want to use out of vocabulary token to index 6, I must set num_words = 7
# because keras count index from 0
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
tokenizer.fit_on_texts(preprocessed_sentences)






