from konlpy.tag import Okt

okt = Okt()


def build_bag_of_words(document: str):

    # remove punctuation
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            bow.insert(len(word_to_index) - 1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            bow[index] = bow[index] + 1

    return word_to_index, bow

from sklearn.feature_extraction.text import CountVectorizer

# must be list of strings
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
# 코퍼스로부터 각 단어의 빈도수를 기록
# how many times each word appears in the document
vector.fit_transform(corpus).toarray()
# how to assign index to each word
vector.vocabulary_


import pandas as pd
from math import log

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
vocab = list(set(w for doc in docs for w in doc.split()))

vocab.sort()

N = len(docs)

def tf(term, d):
    return d.count(term)


def idf(term):
    df = 0
    for doc in docs:
        df += doc.count(term)
    return log(N / (df+1))


def tfidf(t, d):
    return tf(t, d) * idf(t)



result = []

for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)


result = []

for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns=vocab)


result = []

for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])


result = []
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]

vector = CountVectorizer()

vector.fit_transform(corpus).toarray()

vector.vocabulary_

from sklearn.feature_extraction.text import TfidfVectorizer

tfidfv = TfidfVectorizer().fit(corpus)

print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)




















