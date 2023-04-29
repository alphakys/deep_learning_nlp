import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from ydata_profiling import ProfileReport

from nltk.tokenize import word_tokenize

# [STUDY] pandas profiling 예제
# spam_data = pd.read_csv('spam.csv', encoding='latin-1')
# pf = ProfileReport(df=spam_data)
#
# pf.to_file('spam_data.html')

