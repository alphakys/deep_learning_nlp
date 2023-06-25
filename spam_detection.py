import os
import warnings

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

data = pd.read_csv('spam.csv', encoding='latin1')
data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

data['v1'] = data['v1'].replace(['spam', 'ham'], [1, 0])
data.info()
