from utils import linalg
import numpy as np

a = np.matrix([
    [1,1,3],
    [1,5,6],
    [3,6,8],
    [2,3,5]
])

# q, r = linalg.gramSchmidtQR(a)
# print(q * r)
# vals, vectors = linalg.symmetric_eig(a)
# print('eigenvalues =\n{} \neigenvectors =\n{}'.format(vals, vectors))
linalg.svd(a)