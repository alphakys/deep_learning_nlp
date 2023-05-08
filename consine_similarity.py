import numpy as np
from numpy import ndarray
from numpy.linalg import norm

norm()
a = np.array([2, 1, 2, 4, 5])
b = np.array([3, 2, 5, 6, 2])

# make a list of squares with lambda function in list comprehension
[(lambda num: num ** 2)(num) for num in a]


def get_cos_theta(vector_a: ndarray, vector_b: ndarray):

    distance = np.sqrt(np.sum((vector_a - vector_b) ** 2))
    dot = vector_a @ vector_b

    return dot / distance


