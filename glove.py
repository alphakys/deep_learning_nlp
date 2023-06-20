import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

# 동시 등장 확률은 동시 등장 행렬로부터 특정 단어 i의 전체 등장 횟수를 카운트하고, 특정 단어 i가 등장했을 때 어떤 단어 k가
# 등장한 횟수를 카운트하여 계산한 조건부 확률입니다.
#
# P(k | i)
#
# P(context_word | center_word)


