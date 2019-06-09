"""
	Pulls batches of examples from training, validation, or test datasets.
"""

import featureExtractor
import random
import numpy as np
from os import walk

word2index = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
dataPath = './data/'

class TrainingData:
	"""
		Class to load training data with features.
		Stores the examples with features extracted for faster performance.
	"""
	
	def __init__(self, numFeatures):
		self.numFeatures = numFeatures
		# Store as a matrix where each row contains a feature and
		# label vector concatenated
		self.trainingData = np.zeros((0, self.numFeatures + len(word2index.keys())))
		self.loadTrainingData('trainingDataPaths.txt')
	
	def loadTrainingData(self, filePaths):
		"""
			Loads the training data.
		"""
		f = open(filePaths, 'r')
		files = f.readlines()
		for filePath in files:
			featureVector = featureExtractor.getFeatures(filePath[:-1])
			word = filePath.split('/')[2]
			wordIndex = word2index[word]
			# get labels as a one-hot vector
			labelVector = np.zeros(len(word2index.keys()))
			labelVector[wordIndex] = 1
			concatenatedVector = np.concatenate((featureVector, labelVector), axis=0)
			concatenatedVector = concatenatedVector.reshape( \
				(1, self.numFeatures + len(word2index.keys())))
			self.trainingData = np.concatenate((self.trainingData, concatenatedVector), axis=0)
								   
	def getTrainingBatches(self, batchSize, numBatches):
		"""
			Same as getBatches() for the training data but with faster performance.
			For each batch, selects batchSize examples from the data.
			Returns a list of numBatch number of batches that cover the entire
			training dataset.
		"""
		np.random.shuffle(self.trainingData)
		epochXList = []
		epochYList = []
		for i in range(numBatches):
			epochXList.append(self.trainingData[i*batchSize:(i+1)*batchSize,0:self.numFeatures])
			epochYList.append(self.trainingData[i*batchSize:(i+1)*batchSize,self.numFeatures:])
		return epochXList, epochYList

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
		NOTE: now use trainingData.getTrainingBatches for faster performance.
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
