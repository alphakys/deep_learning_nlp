import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)


# dense representation
# 분산 표현을 이용하여 단어 간 의미적 유사성을 벡터화하는 작업ㅇ르 워드 임베딩이라 부르면 이렇게 표현된 벡터를 임베딩 벡터라고 합니다.

# 비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다.

# 윈도우 크기 2

text = ['The', 'fat', 'cat', 'on', 'the', 'mat']

# weights -> units=5 --> multiverse가 5개

# input configuration
time_steps = 32
input_dim = 10

# 임베딩 configuration
embedding_dim = 5

input_shape = (time_steps, input_dim)

# 룩업테이블
projection_layer_weights = (10, 5)


output_dim = 10

# 이유는 마지막 output dense층에서 출력되야하는 dimension은 input_dim과 동일함
# activation function = 'softmax'
projection_output_weights = (1, embedding_dim) * (embedding_dim, input_dim)

input_dim = 7
embedding_dim = 5

# 1. (7, 5)
# 2. (5, 7)




























