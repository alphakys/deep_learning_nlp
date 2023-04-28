
from pathlib import Path

import zipfile, kaggle
# [STUDY] Return a new path with expanded ~ and ~user constructs, as returned by os.path.expanduser(). If a home directory can’t be resolved, RuntimeError is raised.
#   >>>
#   >>> p = PosixPath('~/films/Monty Python')
#   >>> p.expanduser()
#   PosixPath('/home/eric/films/Monty Python')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

dataset_path = Path('./us-patent-phrase-to-phrase-matching')

df = pd.read_csv(dataset_path / 'train.csv')
df.describe(include='object')

df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor

print(df.input.head())










