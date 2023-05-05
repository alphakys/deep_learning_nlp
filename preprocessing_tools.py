import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pykospacing import Spacing


# sent = "니가그렇게띄어쓰기를잘해 내가함 두고두고두고볼게을마나겜을 잘하는 지,과연겜을 잘할까궁금한걸자어디한번 너의실력을펼쳐봐라 ningen"
# spacing = Spacing(sent)

from hanspell import spell_checker
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

sen = "내가너의그오만한코를 납짝하게만들겠다한번 분석해바밬ㅋㅋㅋ"
result = spell_checker.check(sen)

corpus = DoublespaceLineCorpus("2016-10-20.txt")
for document in corpus:
    if len(document) > 0:
        print(document)

word_extractor = WordExtractor(min_frequency=100)
word_score_table = word_extractor.extract()
word_score_table
word_score_table[0]
word_score_table["반포한"]
word_score_table["반포한"].cohesion_forward
word_score_table["반포한강"].cohesion_forward
from soynlp.tokenizer import LTokenizer
scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
scores
l_tokenizer = LTokenizer(scores=scores)
l_tokenizer.tokenize('토크나이징 해주세요 뿌잉뿌잉')
from soynlp.tokenizer import MaxScoreTokenizer
maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
maxscore_tokenizer.tokenize('토크나이징 해주세요 뿌잉뿌잉')
maxscore_tokenizer.tokenize('토크나이징해주세요뿌잉뿌잉')
from soynlp.normalizer import *
emoticon_normalize('개꿀잼ㅋㅋ ㅎ 네 사실 개꿀맛이쥬 ㅋㅋㅋㅋ')
emoticon_normalize('개꿀잼ㅋㅋ ㅎ 네 사실 개꿀맛이쥬 ㅋㅋㅋㅋ', num_repeats=2)

from konlpy.tag import Okt
okt = Okt()
okt.morphs('은경이는 사무실로 갔습니다.')
from ckonlpy.tag import Twitter
twit = Twitter()
twit.add_dictionary('은경이', 'Noun')
twit.morphs('은경이는 사무실로 갔습니다.')
