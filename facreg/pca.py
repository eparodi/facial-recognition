from .methods import Method
import numpy as np
from utils import linalg
from sklearn import svm

class PCA(Method):
    
    def __init__(self):
        super().__init__()
        self.meanImage = None

    def calculate_eigenfaces(self):
        if self.calculated:
            return
        self.calculated = True
        self.meanImage = np.mean(self.trainImages)
        self.trainImages = [self.trainImages[k, :] -
                        self.meanImage for k in range(self.trainImages.shape[0])]
        self.testImages = [self.testImages[k, :] -
                        self.meanImage for k in range(self.testImages.shape[0])]                 
        _, _, self.eigenFaces = linalg.svd(self.trainImages)

        nmax = self.eigenFaces.shape[0]
        accs = np.zeros([nmax, 1])
        for neigen in range(1, nmax+1):
            #Me quedo sólo con las primeras autocaras
            B = self.eigenFaces[0:neigen, :]
            #proyecto
            improy = np.dot(self.trainImages, np.transpose(B))
            imtstproy = np.dot(self.testImages, np.transpose(B))

            #SVM
            #entreno
            self.clf = svm.LinearSVC()
            self.clf.fit(improy, self.trainPerson.ravel())
            accs[neigen-1] = self.clf.score(imtstproy, self.testPerson.ravel())
            print('Precisión con {0} autocaras: {1} %\n'.format(
                neigen, accs[neigen-1]*100))
