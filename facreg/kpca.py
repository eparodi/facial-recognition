import numpy as np
from .methods import Method
from sklearn import svm
from utils import linalg

class KPCA(Method):

    def __init__(self, degree=2):
        super().__init__()
        self.K = None
        self.alpha = None
        self.degree = degree

    def calculate_eigenfaces(self):
        if self.calculated:
            return
        self.calculated = True


        #KERNEL: polinomial de grado degree
        trainNumber = self.trainImages.shape[0]
        testNumber = self.testImages.shape[0]
        self.K = (np.dot(self.trainImages, self.trainImages.T)/ trainNumber + 1)**self.degree
        #K = (K + K.T)/2.0

        #esta transformación es equivalente a centrar las imágenes originales...
        unoM = np.ones([trainNumber, trainNumber])/trainNumber
        self.K = self.K - np.dot(unoM, self.K) - np.dot(self.K, unoM) + np.dot(unoM, np.dot(self.K, unoM))


        #Autovalores y autovectores
        #w,alpha = np.linalg.eigh(K)
        w, self.alpha = linalg.symmetricEig(self.K)
        lambdas = w

        #Los autovalores vienen en orden descendente. Lo cambio
        lambdas = np.flipud(lambdas)
        self.alpha = np.fliplr(self.alpha)

        for col in range(self.alpha.shape[1]):
            self.alpha[:, col] = self.alpha[:, col] / \
                np.sqrt(np.abs(lambdas[col]))

        #pre-proyección
        improypre = np.dot(self.K.T, self.alpha)
        unoML = np.ones([testNumber, trainNumber])/trainNumber
        Ktest = (np.dot(self.testImages, self.trainImages.T)/trainNumber+1)**self.degree
        Ktest = Ktest - np.dot(unoML, self.K) - np.dot(Ktest, unoM) + \
            np.dot(unoML, np.dot(self.K, unoM))
        imtstproypre = np.dot(Ktest, self.alpha)

        nmax = self.alpha.shape[1]
        accs = np.zeros([nmax, 1])
        for neigen in range(1, nmax+1):
            #Me quedo sólo con las primeras autocaras
            #proyecto
            improy = improypre[:, 0:neigen]
            imtstproy = imtstproypre[:, 0:neigen]

            #SVM
            #entreno
            self.clf = svm.LinearSVC()
            self.clf.fit(improy, self.trainPerson.ravel())
            accs[neigen-1] = self.clf.score(imtstproy, self.testPerson.ravel())
            print('Precisión con {0} autocaras: {1} %\n'.format(
                neigen, accs[neigen-1]*100))

    def kerneltrick(self, imagetst):
        trnno = self.trainImages.shape[0]
        tstno = imagetst.shape[0]
        unoM = np.ones([trnno, trnno]) / trnno
        unoML = np.ones([tstno, trnno]) / trnno
        Ktest = (np.dot(imagetst, self.trainImages.T) /
                 trnno + 1) ** self.degree
        Ktest = Ktest - np.dot(unoML, self.K) - np.dot(Ktest, unoM) + \
            np.dot(unoML, np.dot(self.K, unoM))
        imtstproypre = np.dot(Ktest, self.alpha)
        return imtstproypre
