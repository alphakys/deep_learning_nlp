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

movies = pd.read_csv('movies_metadata.csv', low_memory=False)

