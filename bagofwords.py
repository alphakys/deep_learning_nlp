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







{}.get()



