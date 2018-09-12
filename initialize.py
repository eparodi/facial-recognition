from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.externals import joblib

mypath      = '_GRANDE'+'/'
onlydirs    = sorted([f for f in listdir(mypath) if isdir(join(mypath, f))])

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = len(onlydirs)
trnperper   = 6 # trn = training; tst = test
tstperper   = 4
trnno       = personno*trnperper
tstno       = personno*tstperper

