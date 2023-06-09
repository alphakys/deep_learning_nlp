import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd

from string import punctuation

from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.layers import Dense, Embedding, LSTM, SimpleRNN
from keras.models import Sequential


news_pd: pd.DataFrame = pd.read_csv('ArticlesJan2018.csv')

headline_list = news_pd['headline'].tolist()

tk = Tokenizer()
tk.fit_on_texts(headline_list)

vocab_size = len(tk.word_index) + 1

sequences = []
for line in headline_list:
    sequence = tk.texts_to_sequences([line])[0]
    sequences.append(sequence)

encoded_list = []
for seq in sequences[:3]:
    cnt = len(seq)
    for i in range(2, cnt):
        encoded_list.append(seq[:i])

max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(encoded_list)

X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

model = Sequential()

embedding_dim = 10
hidden_units = 10

# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Embedding(output_dim=vocab_size, input_length=)

# [STUDY] inputs에 들어가는 value가 Embedding layer의 input_dim보다 작아야 한다가 핵심이다.
#   np.random.randint(1000, size=(32, 10))에서 의미하는 바는 input의 정수 최댓값은 1000보다 작다는 것을 의미하고
#   embedding layer가 하는 역할은 결국 to_categorical처럼 encoding을 해주는 것인데 만약 들어온 input값이 설정한 vocab_size보다 크면
#   encoding을 할 수가 없다.
#   또한 integer가 1000보다 작아야 한다는 것이다.

inputs = np.random.randint(10, size=(4, 5))
# [STUDY] 중요!!!! 한마디로 embedding의 첫번째 파라미터는 vocab_size을 설정하면 된다.
#   그리고 두번째 파라미터는 output_dim을 설정하면 된다.(Dense의 units와 같은 개념이다.)
e = Embedding(input_dim=10, output_dim=7)


def predict_word(model, tk, search: str, n: int):
    # tokenizer에서 lower로 변환하기 때문에 마찬가지로 lower 함수를 써서 convert한다.
    test_index = tk.word_index[search.lower()]
    # 검색된 리스트 중에서 첫번째를 테스트 sentence로 한다.
    index_arr = [s for s in sequences if test_index in s][0]

    # maxx = len(index_arr)

    # print(index)
















