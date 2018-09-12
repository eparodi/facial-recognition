# -*- coding: utf-8 -*-

from initialize import *
import utils.linalg


#TRAINING SET
images = np.zeros([trnno,areasize])
person = np.zeros([trnno,1])
imno = 0
per  = 0
for dire in onlydirs:
    for k in range(1,trnperper+1):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')
        images[imno,:] = (np.reshape(a,[1,areasize])-127.5)/127.5
        person[imno,0] = per
        imno += 1
    per += 1

#TEST SET
imagetst  = np.zeros([tstno,areasize])
persontst = np.zeros([tstno,1])
imno = 0
per  = 0
for dire in onlydirs:
    for k in range(trnperper,totalperper):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')
        imagetst[imno,:]  = (np.reshape(a,[1,areasize])-127.5)/127.5
        persontst[imno,0] = per
        imno += 1
    per += 1

#KERNEL: polinomial de grado degree
degree = 2
K = (np.dot(images,images.T)/trnno+1)**degree
#K = (K + K.T)/2.0
        
#esta transformación es equivalente a centrar las imágenes originales...
unoM = np.ones([trnno,trnno])/trnno
K = K - np.dot(unoM,K) - np.dot(K,unoM) + np.dot(unoM,np.dot(K,unoM))


#Autovalores y autovectores
#w,alpha = np.linalg.eigh(K)
w,alpha = utils.linalg.symmetricEig(K)
lambdas = w/trnno
lambdas = w

#Los autovalores vienen en orden descendente. Lo cambio 
lambdas = np.flipud(lambdas)
alpha   = np.fliplr(alpha)

for col in range(alpha.shape[1]):
    alpha[:,col] = alpha[:,col]/np.sqrt(np.abs(lambdas[col]))

#pre-proyección
improypre   = np.dot(K.T,alpha)
unoML       = np.ones([tstno,trnno])/trnno
Ktest       = (np.dot(imagetst,images.T)/trnno+1)**degree
Ktest       = Ktest - np.dot(unoML,K) - np.dot(Ktest,unoM) + np.dot(unoML,np.dot(K,unoM))
imtstproypre= np.dot(Ktest,alpha)


nmax = alpha.shape[1]
accs = np.zeros([nmax,1])
for neigen in range(1,nmax+1):
    #Me quedo sólo con las primeras autocaras   
    #proyecto
    improy      = improypre[:,0:neigen]
    imtstproy   = imtstproypre[:,0:neigen]
        
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    accs[neigen-1] = clf.score(imtstproy,persontst.ravel())
    print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[neigen-1]*100))

fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-accs)*100)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Error')

