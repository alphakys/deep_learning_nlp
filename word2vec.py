import os
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml",
#     filename="ted_en-20160408.xml")

targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# sub function은 replace와 같다. 단지 정규식으로 바꾸고자 하는 text를 검색할 뿐
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]

from gensim.models import Word2Vec
from gensim.models import KeyedVectors


#
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)




r = requests.get('I put my google sheets url here')
data = r.text
buffer = io.StringIO(data)

df = pd.read_csv(filepath_or_buffer = buffer, header = 1, usecols = ["Latitude", "Longitude"])




