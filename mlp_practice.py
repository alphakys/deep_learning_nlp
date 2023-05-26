import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import TextVectorization

# Tokenizer에는 list로 각 sentence를 넣어준다.
texts = [['hello'], 'world, nice to meet you', 'hey', '먹고싶은 사과', '먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
tokenizer.texts_to_matrix(texts, mode='tfidf').round(2)

