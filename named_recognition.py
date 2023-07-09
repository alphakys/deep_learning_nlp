import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import re
from nltk import word_tokenize, pos_tag, ne_chunk

# sentence = "James is working at Disney in London"
# tokens = word_tokenize(sentence, language='english')
#
# # 개체명 인식하기 위해서는 pos tagging이 선행되어야 한다.
# tokenized_sentence = pos_tag(tokens)

import urllib

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/train.txt",
#                            filename="recog_train.txt")

f = open('recog_train.txt')
tagged_sentences = []
sentence = []


# for line in f:
#     if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
