import numpy as np
from numpy import matlib


class InvalidMatrixException(Exception):
    pass


def gramSchmidtQR(m: np.matrix):
    """ Returns the matrices of the QR decomposition of the matrix.

    The columns of the matrix should have all its columns linearly independent.

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

    Raises
    ------
    InvalidMatrixException
        If the matrix columns is bigger or equals than the number of rows.
    """
    rows, columns = m.shape
    if columns > rows:
        raise InvalidMatrixException('Invalid size.')
    m = np.float64(m.copy())
    rMatrix = np.matlib.zeros((columns, columns))
    qMatrix = np.matlib.zeros((rows, columns))
    for i in range(columns):
        aux = m[:, i]
        for k in range(i):
            rMatrix[k, i] = qMatrix[:, k].T @ m[:, i]
            aux = aux - rMatrix[k, i] * qMatrix[:, k]
        rMatrix[i, i] = np.linalg.norm(aux)
        qMatrix[:, i] = aux / rMatrix[i, i]
    return qMatrix, rMatrix


def householderQR(m: np.matrix):
    """ Returns the matrices of the complete QR decomposition of the matrix.

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

    Raises
    ------
    InvalidMatrixException
        If the matrix columns is bigger or equals than the number of rows.
    """
    rows, columns = m.shape
    if columns > rows:
        raise InvalidMatrixException('Invalid size.')
    rMatrix = np.copy(m)
    qMatrix = np.eye(rows)
    for i in range(columns - (rows == columns)):
        hMatrix = np.eye(rows)
        xVector = rMatrix[i:, i]
        auxVector = xVector / \
            (xVector[0] + np.copysign(np.linalg.norm(xVector), xVector[0]))
        auxVector[0] = 1
        hMatrix[i:, i:] = np.eye(xVector.shape[0])
        hMatrix[i:, i:] -= (2 / np.dot(auxVector, auxVector)) * \
            np.dot(auxVector[:, None], auxVector[None, :])
        qMatrix = qMatrix @ hMatrix
        rMatrix = hMatrix @ rMatrix
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
        qMatrix, rMatrix = householderQR(eigenValues)
        eigenValues = rMatrix @ qMatrix
        eigenVectors = eigenVectors @ qMatrix

    return eigenValues.diagonal(), eigenVectors


def _sortEig(eigenValues, eigenVectors):
    eigenValues = np.array(eigenValues)
    idx = np.flip(eigenValues.argsort().tolist())
    sortedEigVals = eigenValues[idx]
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
    if rows < columns:
        import ipdb; ipdb.set_trace()
        svdMatrix = m.T @ m
        singValues, vMatrix = symmetricEig(svdMatrix)
        singValues, vMatrix = _sortEig(singValues, vMatrix)
        singValues = np.sqrt(singValues)

        sInvValues = [1/x for x in singValues]
        sInv = np.eye(m.shape[1], m.shape[0])
        np.fill_diagonal(sInv, sInvValues)
        uMatrix = m @ vMatrix @ sInv
        return uMatrix, singValues[:rows], vMatrix.T
    else:
        svdMatrix = m @ m.transpose()
        singValues, uMatrix = symmetricEig(svdMatrix)
        singValues, uMatrix = _sortEig(singValues, uMatrix)
        singValues = np.sqrt(singValues)

        sInvValues = [1/x for x in singValues]
        sInv = np.eye(m.shape[1], m.shape[0])
        np.fill_diagonal(sInv, sInvValues)
        vMatrix = m.T @ uMatrix @ sInv.T
        return uMatrix, singValues[:columns], vMatrix.T

    return uMatrix, singValues, vMatrix
