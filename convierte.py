
from os import listdir
import cv2
import utils.getface

if __name__=="__main__":
        print('Ingrese el nombre de la carpeta')
        maindir=input()
        carpetas=listdir(maindir)
        for carpeta in carpetas:
                archivos=listdir(maindir+carpeta)
                numero=1
                for arch in archivos:
                        cara=utils.getface.getFace(maindir+'/'+carpeta+'/'+arch)
                        if cara is None:
                                continue
                        cv2.imwrite(maindir+'/'+carpeta+'/'+str(numero)+'.pgm',cara)
                        numero+=1

