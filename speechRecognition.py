"""
	Recognizes audio files hopefully!
"""

import featureExtractor
import random
import numpy as np
from os import walk

word2idx = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
dataPath = './data/'

def getBatch(batchSize):
	'''
		This function randomly selects batchSize//5 examples from each
		word's audio files. We also take batchSize%5 examples from a random word's
		audio files to ensure the proper batch size.
		
		TODO: CLEAN CODE
		Create lists of training and testing filepaths.
		Are batches independent?
	'''
	epochX = []
	epochY = np.zeros(batchSize)
	count = 0

	for word in word2idx.keys():
		files = []
		for (dirPath, dirNames, fileNames) in walk(dataPath + word + '/'):
			files.extend(fileNames)
		sampleFiles = random.sample(files, batchSize//5)
		for fileName in sampleFiles:
			if fileName[0:2] == "._":
				fileName = fileName[2:]
			epochX.append(featureExtractor.getFeatures(dataPath + word + '/' + fileName))
			epochY[count] = word2idx[word]
			count += 1
	
	# get files to fill in the rest of the batch
	files = []
	word = random.sample(word2idx.keys(),1)[0]
	for (dirPath, dirNames, fileNames) in walk(dataPath + word + '/'):
		files.extend(fileNames)
	sampleFiles = random.sample(files, batchSize%5)
	for fileName in sampleFiles:
		if fileName[0:2] == "._":
			fileName = fileName[2:]
		epochX.append(featureExtractor.getFeatures(dataPath + word + '/' + fileName))
		epochY[count] = word2idx[word]
		count += 1
	
	return np.array(epochX), epochY.reshape(100, 1)

def getTestData(batchSize):
	# TODO: make test data
	return getBatch(batchSize)
