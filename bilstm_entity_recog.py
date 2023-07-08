import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
# data = pd.read_csv("ner_dataset.csv", encoding="latin1")

data = pd.read_csv('ner_dataset.csv', encoding='latin1')

