"""
	Pulls batches of examples from training, validation, or test datasets.
"""

import featureExtractor
import random
import numpy as np
from os import walk

word2index = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
dataPath = './data/'

def getTotalNumFiles(filePaths = 'trainingDataPaths.txt'):
	"""
		Returns the number of filepaths listed in a text file.
		By default returns the number of training data paths.
	"""
	f = open(filePaths, 'r')
	return len(f.readlines())

def getBatches(batchSize, numBatches, filePaths = 'trainingDataPaths.txt'):
	"""
		For each batch, selects batchSize examples from the data.
		Returns a list of numBatch number of batches that cover the entire
		training dataset.
	"""
	f = open(filePaths, 'r')
	files = f.readlines()
	random.shuffle(files)
	
	epochXList = []
	epochYList = []
	for i in range(numBatches):
		epochX = []
		epochY = np.zeros(batchSize)
		count = 0
		sampleFiles = files[i*batchSize:(i+1)*batchSize]
		for filePath in sampleFiles:
			# get rid of the new line character
			epochX.append(featureExtractor.getFeatures(filePath[:-1]))
			word = filePath.split('/')[2]
			epochY[count] = word2index[word]
			count += 1
		# get y labels as one-hot vectors
		epochY_one_hot = np.zeros((batchSize, len(word2index.keys())))
		for i in range(len(epochY)):
			index = int(epochY[i])
			epochY_one_hot[i][index] = 1

		epochXList.append(np.array(epochX))
		epochYList.append(epochY_one_hot)
	return epochXList, epochYList

def getTestData():
	"""
		Returns the test data batch.
	"""
	testSize = getTotalNumFiles('testDataPaths.txt')
	testX, testY = getBatches(testSize, 1, filePaths = 'testDataPaths.txt')
	return testX[0], testY[0]

def getValidationData():
	"""
		Returns the validation data batch.
	"""
	validationSize = getTotalNumFiles('validationDataPaths.txt')
	valX, valY = getBatches(validationSize, 1, filePaths = 'validationDataPaths.txt')
	return valX[0], valY[0]
