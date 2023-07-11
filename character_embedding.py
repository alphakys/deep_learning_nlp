import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer

data = pd.read_csv('ner_dataset.csv', encoding='latin1')

data = data.fillna(method="ffill")

data['Word'] = data['Word'].str.lower()

func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences = [t for t in data.groupby("Sentence #").apply(func)]

words = list(set(data['Word'].values))
chars = set(''.join(words))
list(chars)
# chars = sorted(list(chars))

words = list(set(data['Word'].values))
chars = set(''.join(words))
chars = sorted(list(chars))

[int(c) for c in chars]
[int(bin(c)) for c in chars]
[ord(c) for c in chars]
char_to_index = {v: k + 2 for k, v in enumerate(chars)}
char_to_index
char_to_index['OOV'] = 1
char_to_index['PAD'] = 0
chars
index_to_char = {v: k for k, v in char_to_index}
index_to_char = {v: k for k, v in char_to_index.items()}
index_to_char
# 문자 padding length = 15

max_len = 15

max_len
from keras.utils import pad_sequences

words[:10]
for w in words[:10]:
    print(w)

for w in words[:10]:
    print(w.zfill(max_len))

for w in words[:10]:
    pad_sequences(w, max_len)

for w in words[:10]:
    padded_char = pad_sequences([char_to_index[c] for c in w], max_len)
    print(padded_char)

for w in words[:10]:
    # padded_char = pad_sequences([char_to_index[c] for c in w], max_len)

    print([char_to_index[c] for c in w])

char_to_index['s']
char_to_index['p']
[[char_to_index[c] for c in w] for w in words]
padded_seq = pad_sequences([[char_to_index[c] for c in w] for w in words], maxlen=max_len)
padded_seq
padded_seq[0]
max_len_char = 15


# 문자 시퀀스에 대한 패딩하는 함수
def padding_char_indice(char_indice, max_len_char):
    return pad_sequences(
        char_indice, maxlen=max_len_char, padding='post', value=0)


# 각 단어를 문자 시퀀스로 변환 후 패딩 진행
def integer_coding(sentences):
    char_data = []
    for ts in sentences:
        word_indice = [word_to_index[t] for t in ts]
        char_indice = [[char_to_index[char] for char in t]
                       for t in ts]
        char_indice = padding_char_indice(char_indice, max_len_char)

        for chars_of_token in char_indice:
            if len(chars_of_token) > max_len_char:
                continue
        char_data.append(char_indice)
    return char_data


# 문자 단위 정수 인코딩 결과
X_char_data = integer_coding(sentences)

sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:  # 47,959개의 문장 샘플을 1개씩 불러온다.

    # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentence, tag_info = zip(*tagged_sentence)
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info))  # 각 샘플에서 개체명 태깅 정보만 저장한다.

sentences
sentences[:10]
[[word for word in sen] for sen in sentences[:10]]
sentences[0]
len(sentences[0])