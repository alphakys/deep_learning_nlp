import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import urllib.request
from matplotlib import pyplot as plt
from collections import Counter
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename='steam.txt')

# read_table can read tab separated data and names -> coumns name
data = pd.read_table('steam.txt', names=['label', 'reviews'])

# number of unique data
data['reviews'].nunique()
# duplicated indicate whether the data is duplicated or not by boolean[True, False]
data['reviews'].duplicated()

remove_duplicate = data['reviews'].drop(data['reviews'][data['reviews'].duplicated()].index)
# data.drop_duplicates(subset=['reviews'], inplace=True)
data['reviews'] = remove_duplicate

