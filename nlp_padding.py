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

pad_sequences(encoded, maxlen=7, dtype='int8')
