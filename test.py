import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('ronfic_u_xim.csv')
df: DataFrame = df.loc[:, df.columns != '#']

model = KMeans(n_clusters=2, algorithm='auto')
model.fit(df)

df['prediction'] = model.predict(df)

# print(df[['users_id', 'prediction']].to_csv('result.csv'))


# print(df.groupby(by=['users_id', 'prediction']).count())

exit()

plt.scatter(df['R_avr_force'], df['weight_value'], c=df['prediction'], alpha=0.5)


# 로짓함수 시그모이드 함수에 대한 전반적인 이해를 위한 수학공부!
plt.show()

exit()

# [STUDY]
#  C order란 ROW MAJOR ORDER를 의미한다.
#  F order란 COLUMN MAJOR ORDER를 의미한다.
n_arr = np.ndarray(shape=(10, 10), dtype=int, buffer=np.array(range(10)), order='C', offset=1)


# [STUDY]
#   결국에는 vectorization이란 cpu가 수행할 명령어 세트를 instructon이라고 하면 이 instruction을 수행할 데이터가
#   하나 by 하나라면 시간이 오래 걸릴 것인데
#   modern cpu는 하나의 instruction을 처리하는데 여러 multiple data를 가지고 와서도 처리할 수 있게 되었다는 점
#   register에 담을 수 있는 operand 그릇이 여러개니까 여러 데이터를 가져와서 한 명령어에서 수평적으로 처리할 수 있게 한다
#   그리고 numpy에서는 결국 이와 같은 연산을 수행하기 위해서 만약에 길이가 다른 두 array가 있다면 vectorization을 시키기 위해서
#   작은 array의 길이를 copy하여 길이를 맞춰 주고 그릇에 여러 데이터를 한 번에 담아서 operation을 수행한다는 점.
#    ** VECTORIZATION **
#     The software changes required to exploit instruction level parallelism are know ans vectorization
#    Single Instruction Multiple Data
#    ** pipe line **
#   유닉스 계열 운영 체제에서(어느 정도까지는 마이크로소프트 윈도우에서) ㅔㅈ공되는 병행성 매커니즘의 하나로서, 두 프로세스가 생산자-소비자 모델에 따라 통신할 수 있게 해주는 원형
#   버퍼이다. 즉 파이프는 한 프로세스가 쓰고 다른 프로세스가 읽는 선입선출 형태의 큐라 할 수 있다. 파이프의 개념은 코루틴으로부터 영향을 받아 만들어졌으며, ㅎ운영 체제 기술의 발전에 큰 공헌을
#   print(하였다)
#    파이프에는 일정한 크기의 공간이 할당되어 있다. 어떠 ㄴ프로세스가 파이프에 데이터를 기록하려고 할 때, 충분한 공간이 남아있다면 기록이 즉시 수행되겠지만, 공간이 부족하다면 그 프로세스는
#    print(차단된다)이것은 웅여체제가 상호배제를 수행한 결과이다.
