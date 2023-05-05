import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'],
                          ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'],
                          ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
                          ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
                          ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)

padded = pad_sequences(encoded, maxlen=7, dtype='int8')

from konlpy.tag import Okt

okt = Okt()

txt = '나는 자연어 처리를 배운다.'

okt.morphs(txt)

sub_text = "점심 먹으러 갈래 메뉴는 햄버거가 최고야"
# the argument of texts_to sequences must be list type
# texts~ 함수의 argument는 리스트가 되어야 하는것 같다.
encoded = tokenizer.texts_to_sequences([sub_text])
from keras.utils import to_categorical
# texts_to_sequences를 통해 각 단어가 integer encoding된 argument를 keras to_categorical 함수에 인자로 넣어준다.
# must input the argument of to_categorical function's return value
one_hot = to_categorical(encoded)
one_hot

















