from utils import linalg
import numpy as np

a = np.matrix([
    [1,1,3],
    [1,5,6],
    [3,6,8],
    [2,3,5]
])

u, s, v = linalg.svd(a)
print( '\
        u = \n{}\n\
        s = \n{}\n\
        v = \n{}'.format(u,s,v))
sMatrix = np.eye(a.shape[0], a.shape[1])
np.fill_diagonal(sMatrix, s.A1)
print(u * sMatrix * v.T)
