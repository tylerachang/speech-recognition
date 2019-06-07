"""
	Recognizes audio files hopefully!
"""

import featureExtractor
import random
import numpy as np
from os import walk

word2idx = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
dataPath = './data/'

def getBatch(batchSize, numBatch, fileName = 'trainingDataPaths.txt'):
	'''
		This function randomly selects `batchSize` examples from the training
		data, and it returns a list of `numBatch` number of batches that cover
		the whole training dataset.
	'''
	f = open(fileName, 'r')
	files = f.readlines()
	random.shuffle(files)

	epochXList = []
	epochYList = []
	for i in range(numBatch):
		epochX = []
		epochY = np.zeros(batchSize)
		count = 0
		sampleFiles = files[i*batchSize:(i+1)*batchSize]
		for filePath in sampleFiles:
			# get rid of the new line character
			epochX.append(featureExtractor.getFeatures(filePath[:-1]))
			word = filePath.split('/')[2]
			epochY[count] = word2idx[word]
			count += 1
		epochXList.append(np.array(epochX)), epochYList.append(epochY.reshape(100, 1))

	return epochXList, epochYList

def getTestData(batchSize):
	'''
		This function randomly selects batchSize examples from the test data.
	'''
	return getBatch(batchSize, 1, fileName = 'testDataPaths.txt')
