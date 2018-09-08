import numpy as np
from numpy import matlib

class InvalidMatrixException(Exception):
    pass

def gramSchmidtQR(m: np.matrix):
    """ Returns the matrices of the QR decomposition of the matrix.

    Parameters
    ----------
    m: np.matrix
        The matrix to decompose.

    Returns
    -------
    np.matrix:
        The matrix Q of the decomposition. This is an orthonormal matrix.    
    np.matrix:
        The matrix R of the decomposition. This is an upper triangular
        matrix.

    """
    rows, columns = m.shape
    qMatrix = np.matlib.zeros((rows, rows))
    qMatrix[:,0] = m[:, 0] / np.linalg.norm(m[:, 0], 2)
    for i in range(1, columns):
        qColumn = m[:, i]
        for j in range(0, i):
            aux = np.dot(qMatrix[:, j].transpose(), m[:, i])
            aux = aux[0,0] * qMatrix[:, j]
            qColumn = qColumn - aux
        qMatrix[:, i] = qColumn / np.linalg.norm(qColumn, 2)
    rMatrix = np.transpose(qMatrix) @ m
    return qMatrix, rMatrix


def symmetricEig(m: np.matrix):
    """ Returns the eigenvalues and eigenvectors of a symmetric matrix.

    This function calculates the eigenvalues using the QR method. Then the
    eigenvectors are calculated using a recursion of this method that only
    works for symmetric matrix. So it can be used to get the eigenvalues
    of any matrix.

    Parameters
    ----------
    m: np.matrix
        The matrix to find the eigenvalues and eigenvectors.

    Returns
    -------
    np.matrix
        A matrix with the eigenvalues of a in the diagonal.
    np.matrix
        A matrix with the eigenvectors of a as columns.

    Raises
    ------
    InvalidMatrixException
        If the matrix is not squared.

    """
    if (m.shape[0] != m.shape[1]):
        raise InvalidMatrixException('The matrix is not squared.')
    
    dimension = m.shape[0]
    eigenValues = m
    eigenVectors = np.eye(dimension)
    for _ in range(0, 1000):
        qMatrix, rMatrix = gramSchmidtQR(eigenValues)
        eigenValues = rMatrix @ qMatrix
        eigenVectors = eigenVectors @ qMatrix

    return eigenValues.diagonal(), eigenVectors


def _sortEig(eigenValues, eigenVectors):
    idx = np.flip(eigenValues.argsort().tolist())[0]
    sortedEigVals = eigenValues[:, idx]
    sortedEigVecs = eigenVectors[:, idx]
    return sortedEigVals, sortedEigVecs


def svd(m: np.matrix):
    """ Returns the SVD of a matrix.

    Parameters
    ----------
    m: np.matrix
        The matrix to decompose

    Returns
    -------
    np.matrix or None
        The first matrix of the decomposition. This is a 
        orthogonal matrix.
    np.matrix
        The matrix with the singular values in the diagonal.
    np.matrix
        The second matrix of the decomposition. This one is
        also a orthogonal matrix.
    """
    rows, columns = m.shape
    if rows > columns:
        svdMatrix = m.transpose() @ m
        singValues, vMatrix = symmetricEig(svdMatrix)
        singValues, vMatrix = _sortEig(singValues, vMatrix)
        singValues = np.sqrt(singValues)

        return None, singValues, vMatrix
    else:
        svdMatrix = m * m.transpose()
        singValues, uMatrix = symmetricEig(svdMatrix)
        singValues, uMatrix = _sortEig(singValues, uMatrix)
        singValues = np.sqrt(singValues)

        return singValues, vMatrix, None
    
    return uMatrix, singValues, vMatrix
