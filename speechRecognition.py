"""
	Recognizes audio files (.wav format) using neural networks.
"""

import neuralNetworks as nn
import recurrentNeuralNetworks as rnn
import datetime

#rnn = rnn.RecurrentNeuralNetwork(5, 50, 10, 128, 100)
#rnn.train_neural_network(5, "")
#rnn.evaluate_model("x")
#neuralNetwork = nn.NeuralNetwork(5, 500, 32, [100,100,100])
#neuralNetwork.train_neural_network(10, "b")
#neuralNetwork.evaluate_model("testModel")

def testNumNodes():
	numEpochs = 40
	# NN tests, using batch size 32, 3 hidden layers
	numNodes = 2
	for i in range(9):
		print("NN: numNodes=", numNodes)
		neuralNetwork = nn.NeuralNetwork(5, 500, 32, [numNodes,numNodes,numNodes])
		startTime = datetime.datetime.now()
		neuralNetwork.train_neural_network(numEpochs, "")
		timeToRun = datetime.datetime.now() - startTime
		numNodes *= 2
		print("Time to train: ", timeToRun)
	# RNN tests, using batch size 32
	numNodes = 2
	for i in range(9):
		print("RNN: numNodes=", numNodes)
		recurrentNN = rnn.RecurrentNeuralNetwork(5, 50, 10, numNodes, 32)
		startTime = datetime.datetime.now()
		recurrentNN.train_neural_network(numEpochs, "")
		timeToRun = datetime.datetime.now() - startTime
		numNodes *= 2
		print("Time to train: ", timeToRun)

def testNeuralNet():
	# using batch size 32, 3 hidden layers, 128 nodes per hidden layer
	neuralNetwork = nn.NeuralNetwork(5, 500, 32, [128,128,128])
	# 1000 epochs
	neuralNetwork.train_neural_network(1000, "NN")
	
def testRecurrentNeuralNet():
	# using batch size 32, 128 nodes per layer
	recurrentNN = rnn.RecurrentNeuralNetwork(5, 50, 10, 128, 32)
	# 1000 epochs
	recurrentNN.train_neural_network(1000, "RNN")

#testNumNodes()
#testNeuralNet()
testRecurrentNeuralNet()