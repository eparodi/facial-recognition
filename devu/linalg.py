# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib as matlib

def eig(a):
    """
    Receives a numpy.matrix.
    Returns (eigenvalues, eigenvectors)
    """
    x = a
    rows = a.shape[0]
    columns = a.shape[1]
    if (rows != columns):
        raise 'Rows and Columns does not match!'

    s = np.eye(rows)
    for i in range(0,1000):
        q,r = qrgs(x)
        x = r @ q
        s = s @ q

    # Fill with zeros in the superior and inferior triangles
    for i in range(0, rows):
        for j in range(0, columns):
            if (i != j):
                x[i,j] = 0

    for i in range(0, rows):
        s[:,i] = s[:,i] / np.linalg.norm(s[:,i], 2)

    return (x,s)

def qrgs(a):
    """
    Receives a numpy.matrix.
    Returns (q,r)
    """
    rows = a.shape[0]
    columns = a.shape[1]
    q_m = matlib.zeros((rows, rows))
    q_m[:,0] = a[:,0] / np.linalg.norm(a[:,0],2)
    for i in range(1, columns):
        pi = a[:,i]
        for j in range(0, i):
            pi = pi - np.dot(q_m[:,j].transpose(), a[:,i])[0,0] * q_m[:,j]
        q_m[:,i] = pi / np.linalg.norm(pi, 2)
    r = np.transpose(q_m) * a
    return (q_m,r)


def svd(a):
    m = a * a.transpose()
    sval1, u = eig(m)
    m = a.transpose() * a
    sval2, v = eig(m)
    sigma = np.sqrt(sval2)
    sigma = np.vstack([sigma, [0,0,0]])
    u = _orthonormalize(u)
    v = _orthonormalize(v)
    # print(u * sigma * v.transpose())
    # print(u * sval2 * v.transpose())
    # print(v * sval1 * u.transpose())
    # print(v * sval2 * u.transpose())
    # print('{}\n{}\n{}\n{}'.format(sigma,u,sigma,v))


def _orthonormalize(a):
    n = a.shape[0]
    q_m = matlib.zeros((n, n))
    # import ipdb; ipdb.set_trace()
    q_m[:, 0] = a[:, 0] / np.linalg.norm(a[:, 0], 2)
    for i in range(1, n):
        pi = a[:, i]
        for j in range(0, i):
            pi = pi - np.dot(q_m[:, j].transpose(), a[:, i])[0, 0] * q_m[:, j]
        q_m[:, i] = pi / np.linalg.norm(pi, 2)
    return q_m

# (q,r) = qrgs(np.matrix([[1,2,3],[2,2,4],[5,6,7]]))
# print('q: {q}\nr: {r}'.format(q=q, r=r))
a = np.matrix([
        [1, 2, 3],
        [2, 2, 4],
        [5, 6, 7],
        [3, 4, 6]
    ])
svd(a)
# eigval, eigvect = eig(a)
# print(eigval)
# print(eigvect)
# print(a)
# x = eigvect[:,1]

# print(a*x)
# print(eigvalk[1,1] * x)
