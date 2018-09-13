import os
from facreg import KPCA, PCA, WebCamInterface

def get_method():
    OPTIONS = [
        1, # PCA
        2, # KPCA
    ]
    finish = False
    while not finish:
        print('Elija un método:\n1 > PCA\n2 > KPCA')
        try:
            option = int(input())
            if not option in OPTIONS:
                raise Exception()
            if option == 1:
                print('>> PCA')
                return PCA()
            else:
                print('>> KPCA')
                return KPCA()
            finish = True
        except Exception as e:
            print(e)
            print('La opción no es válida.')

def main():
    method = get_method()
    method.calculate_eigenfaces()
    webcam = WebCamInterface(method)
    webcam.start_webcam()

if __name__ == '__main__':
    main()
