from os import listdir
from os.path import isdir, join
import json
from scipy import ndimage as im
import numpy as np

class Method():

    def __init__(self):
        self.calculated = False
        self.eigenFaces = []
        self.clf = None
        with open('config.json') as f:
            config = json.load(f)
        self.config = config
        areaSize = config['imageWidth'] * config['imageHeight']
        sets = sorted([f for f in listdir(config['path']) if isdir(join(config['path'], f))])
        peopleNumber = len(sets)
        imageNumber = 0
        personNumber = 0
        totalImages = config['testImages'] + config['trainImages']
        # Loading Training Set
        trainNumber = peopleNumber * config['trainImages']
        self.trainImages = np.zeros([trainNumber, areaSize])
        self.trainPerson = np.zeros([trainNumber, 1])
        for dire in sets:
            for k in range(1, config['trainImages']+1):
                a = im.imread(join(config['path'], dire +'/{}'.format(k) + '.pgm'))/255.0
                self.trainImages[imageNumber, :] = np.reshape(a, [1, areaSize])
                self.trainPerson[imageNumber, 0] = personNumber
                imageNumber += 1
            personNumber += 1

        # Loading Testing Set
        testNumber = peopleNumber * config['testImages']
        self.testImages = np.zeros([testNumber, areaSize])
        self.testPerson = np.zeros([testNumber, 1])
        imageNumber = 0
        personNumber = 0
        for dire in sets:
            for k in range(config['trainImages'], totalImages):
                a = im.imread(join(config['path'], dire + '/{}'.format(k) + '.pgm'))/255.0
                self.testImages[imageNumber, :] = np.reshape(a, [1, areaSize])
                self.testPerson[imageNumber, 0] = personNumber
                imageNumber += 1
            personNumber += 1
