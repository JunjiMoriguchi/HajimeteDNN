import numpy as np

index = np.arange(10)
print(index)
print(index[:5])
print(index % 3 == 0)
index_hoge = index[index%3==0]
print(index_hoge)