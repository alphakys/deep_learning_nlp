import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import urllib.request
from matplotlib import pyplot as plt
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename='steam.txt')

