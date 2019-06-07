"""
	Recognizes audio files hopefully!
"""

import neuralNetworks
import featureExtractor
import random
import numpy as np
from os import walk

word2idx = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
myPath = './data.02/'

def getBatch(batchSize):
	'''
		This function randomly select batchSize/5 number of data from each
		word's audio files. We take batchSize/5 + remainder from the last
		word's audio file.
	'''
	epochX = []
	epochY = np.zeros(batchSize)
	count = 0

	for word in word2idx.keys():
		files = []
		for (dirPath, dirNames, fileNames) in walk(myPath + word + '/'):
			files.extend(fileNames)
		sampleFiles = random.sample(files, batchSize//5)
		for fileName in sampleFiles:
			epochX.append(featureExtractor.getFeatures(fileName))
			epochY[count] = word2idx[word]
			count += 1

	files = []
	word = random.sample(word2idx.keys(),1)[0]
	for (dirPath, dirNames, fileNames) in walk(myPath + word + '/'):
		files.extend(fileNames)
	sampleFiles = random.sample(files, batchSize%5)
	for fileName in sampleFiles:
		epochX.append(featureExtractor.getFeatures(fileName))
		epochY[count] = word2idx[word]
		count += 1

	print(epochX, epochY)
	return epochX, epochY

def getTestData():
	pass
