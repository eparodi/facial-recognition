from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib

mypath      = '_GRANDE2'+'/'
onlydirs    = sorted([f for f in listdir(mypath) if isdir(join(mypath, f))])

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = len(onlydirs)
trnperper   = 14 # trn = training; tst = test
tstperper   = 6
totalperper = tstperper +trnperper
trnno       = personno*trnperper
tstno       = personno*tstperper

