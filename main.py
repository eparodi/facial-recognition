from utils import linalg
import numpy as np

a = np.matrix([
    [1,1,3],
    [1,5,6],
    [3,6,8],
    [2,3,5]
])

_, s, v = linalg.svd(a)
print(s)
print(v)
