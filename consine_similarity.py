import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np


import pandas as pd
from numpy import ndarray
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# l1-norm 절대값 거리 // l2-norm 유클리디언 거리(일반적으로 알고 있는 좌표평면 상의 두점 사이의 거리)
# l2-norm 단 원점에서의 거리를 말함

# 코사인 유사도란 결국 두 벡터의 내적을 알고 있을 때, cosine 값을 알 수 있다는 것을 이용하는 방법인데
# 코사인은 결국 두 벡터 혹은 여러 벡터 사이의 각도?를 가지고 그 유사도를 비교하는 방식을 말한다.
def get_cos_theta(vector_a: ndarray, vector_b: ndarray):

    dot = np.dot(vector_a, vector_b)
    vector_distance = norm(vector_a) * norm(vector_b)

    return dot / vector_distance

# movies = pd.read_csv('movies_metadata.csv', low_memory=False)
# movies pandas에 Nan 값을 '' 대체해서 채운다.
# movies = movies.fillna('')
#
# TfidfVectorizer(stop_words='english') => 불용어 initialize할 때 설정
# tfidf = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf.fit_transform(movies['overview'])


# combined_cosine = sorted(list(zip(title_to_index.keys(), cosine_sim[0])), key=lambda x: x[1])[::-1]


# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# print(cosine_sim.shape)

pd_movies = pd.read_csv('movies_metadata.csv', low_memory=False)
pd_movies['overview'].fillna('', inplace=True)
sample_pd = pd_movies[:1000]

title_to_index = {v: idx for idx, v in enumerate(sample_pd['title'])}

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(sample_pd['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

key_arr = title_to_index.keys()
sim_result = np.ones((1,998,2))
for cosine in cosine_sim:
    tmp = sorted(list(zip(key_arr, cosine)), key=lambda x: x[1])[::-1]
    tmp = np.array(tmp)
    tmp = np.expand_dims(tmp, axis=0)
    sim_result = np.append(sim_result, tmp, axis=0)

# [STUDY] 3차원 행렬의 0번째 차원을 제거하는 방법
sim_result = np.delete(sim_result, 0, axis=0)


# 복습
# 코사인 유사도라 함은 두 벡터 사이의 각도를 이용하여 얼마나 유사한지를 측정하는 방법이다.
# 언뜻 좌표선상에서 서로 비슷한 두 좌표를 어떻게 계산할 것인가를 물어본다면 두 좌표 사이의 거리를 말할 것이다.
# 하지만 코사인 유사도에서 똑같다고 말하는 두 좌표를 생각해보자 두 좌표는 같은 방향을 가리키고 있지만 거리상으로 멀리 떨어져 있다고 가정해보자
# 그러면 좌표 사이의 거리를 가지고 측정한 유사도에서는 두 document는 굉장히 차이를 가지지만 사실은 같은 방향을 가리키는 유사한 문서인 것이다.
# 이와같이 두 벡터 사이의 각도를 이용하여 유사도를 측정하는 방법을 코사인 유사도라고 한다.
























