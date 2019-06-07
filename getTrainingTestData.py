"""
    Generate two files for filepaths of training and test data.
"""
import random
from os import walk

words = ['backward','bed', 'bird','cat', 'dog']
dataPath = './data.02/'

def getAllFiles():
    '''
        Returns a list of all file names (paths) of the audio files of words
        that we care about
    '''
    files = []
    for word in words:
        for (dirPath, dirNames, fileNames) in walk(dataPath + word + '/'):
            for fileName in fileNames:
                if fileNames[0] != '.': # do not include hidden files
                    files.append(dataPath + word + '/' + fileName)
    return files

def getData(percentOfTraining):
    '''
        Generate training data and test data paths based on the words and
        percentage of data that are desired to be for training.
    '''
    allFiles = getAllFiles()
    random.shuffle(allFiles)
    numTraining = int(len(allFiles) * percentOfTraining)
    trainingList = allFiles[:numTraining+1]
    testList = allFiles[numTraining+1:]
    # potentially return here and call this function in neuralNetworks.py

    # write the file names to textfiles
    trainingFile = open("trainingDataPaths.txt","w+")
    for fileName in trainingList:
        trainingFile.write(fileName+"\n")
    trainingFile.close()
    testFile = open("testDataPaths.txt","w+")
    for fileName in testList:
        testFile.write(fileName+"\n")
    testFile.close()

getData(0.8)
