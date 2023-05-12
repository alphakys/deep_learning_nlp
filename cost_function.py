import warnings

# [STUDY] PANDAS FUTURE WARNINGS
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

house_d = pd.read_csv('boston.csv')

print(house_d.head(5))

X_house = house_d['DIS']
y_true = house_d['PRICE']

X = np.array(X_house).reshape(-1, 1)
ss = StandardScaler()
ss_X = ss.fit_transform(X)

print(ss_X)




# plt.plot(ss_X, y_true, 'ro')
# plt.show()


def hypothesys():
    ...
