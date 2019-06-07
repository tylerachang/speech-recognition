"""
	Recognizes audio files hopefully!
"""

import featureExtractor
import random
import numpy as np
from os import walk

word2idx = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
dataPath = './data/'

def getBatch(batchSize, fileName = 'trainingDataPaths.txt'):
	'''
		This function randomly selects batchSize examples from the training data

		TODO: CLEAN CODE
		Are batches independent?
	'''
	epochX = []
	epochY = np.zeros(batchSize)
	count = 0

	f = open(fileName, 'r')
	files = f.readlines()

	sampleFiles = random.sample(files, batchSize)
	for filePath in sampleFiles:
		# get rid of the new line character
		epochX.append(featureExtractor.getFeatures(filePath[:-1]))
		word = filePath.split('/')[2]
		epochY[count] = word2idx[word]
		count += 1

	return np.array(epochX), epochY.reshape(100, 1)

def getTestData(batchSize):
	'''
		This function randomly selects batchSize examples from the test data
	'''
	return getBatch(batchSize, fileName = 'testDataPaths.txt')
