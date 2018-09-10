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
b=a*a.transpose()
w1,l1=np.linalg.eig(b)
w2,l2=linalg.symmetricEig(b)
print("con numpy EIG")
print(sorted(list(w1)))
print(l1)
print("con la de casera")
print(sorted(list(np.asarray(w2)[0])))
print(l2)