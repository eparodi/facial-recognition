import unittest
import numpy as np

import linalg

class LinearAlgebraTests(unittest.TestCase):

    ERROR = 0.001

    MATRIX_1 = np.matrix(
        [
            [1, 2, 3],
            [2, 4, 6],
            [3, 5, 7]
        ]
    )

    MATRIX_2 = np.matrix(
        [
            [1, 9, 1],
            [5, 2, 4],
            [6, 3, 0],
            [9, 4, 2]
        ]
    )

    MATRIX_3 = MATRIX_2.T

    matrices = [MATRIX_1, MATRIX_2, MATRIX_3]

    def check_simmilarity(self, m1, m2):
        if (m1.shape != m2.shape):
            return False
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                if np.isnan(m1[i,j]) or np.isnan(m2[i,j]):
                    return False
                if abs( abs(m1[i, j]) - abs(m2[i, j]) ) > self.ERROR:
                    return False
        
        return True


    def check_list(self, l1:list, l2:list):
        l1 = sorted(l1.copy())
        l2 = sorted(l2.copy())
        if (len(l2) != len(l1)):
            return False
        for i in range(len(l1)):
            if abs( abs(l1[i]) - abs(l2[i]) ) > self.ERROR:
                return False
        
        return True


    def test_gramSchmidtQR(self):
        with self.assertRaises(linalg.InvalidMatrixException):
            linalg.gramSchmidtQR(self.MATRIX_2.T)
        q, r = np.linalg.qr(self.MATRIX_2)
        q2, r2 = linalg.gramSchmidtQR(self.MATRIX_2)

        # Check if the decomposition is correct
        self.assertTrue(self.check_simmilarity(q2 @ r2, self.MATRIX_2))
        # Check if q is orthogonal
        identity = np.eye(q2.shape[1], q2.shape[1])
        self.assertTrue(self.check_simmilarity(q2.T @ q2, identity))
        # Check q and r values with another library
        self.assertTrue(self.check_simmilarity(q, q2))
        self.assertTrue(self.check_simmilarity(r, r2))


    def test_householderQR(self):
        for matrix in self.matrices:
            rows, columns = matrix.shape
            if columns > rows:
                with self.assertRaises(linalg.InvalidMatrixException):
                    linalg.gramSchmidtQR(matrix)
                continue
            q, r = np.linalg.qr(matrix, 'complete')
            q2, r2 = linalg.householderQR(matrix)
            # Check if the decomposition is correct
            self.assertTrue(self.check_simmilarity(q2 @ r2, matrix))
            # Check if q is orthogonal
            identity = np.eye(q2.shape[1], q2.shape[1])
            self.assertTrue(self.check_simmilarity(q2.T @ q2, identity))
            # Check q and r values with another library
            self.assertTrue(self.check_simmilarity(q, q2))
            self.assertTrue(self.check_simmilarity(r, r2))


    def test_eig(self):
        testMatrix = self.MATRIX_2.T * self.MATRIX_2
        eigVal, _ = np.linalg.eig(testMatrix)
        eigVal2, eigVec2 = linalg.symmetricEig(testMatrix)
        
        # Check eigenvalues
        self.assertTrue(self.check_list(eigVal, eigVal2))
        eigVal2 = np.array(eigVal2)

        # Check eigenvectors
        for i in range(eigVec2.shape[1]):
            vector = eigVec2[:,i]
            firstValue = testMatrix @ vector
            secondValue = np.matrix(eigVal2[i] * vector)
            self.assertTrue(self.check_simmilarity(firstValue, secondValue))


if __name__ == '__main__':
    unittest.main()
