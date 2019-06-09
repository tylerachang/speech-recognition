"""
	Recognizes audio files (.wav format) using neural networks.
"""

import neuralNetworks as nn
import recurrentNeuralNetworks as rnn
import datetime

#rnn = rnn.RecurrentNeuralNetwork(5, 50, 10, 128, 100)
#rnn.train_neural_network(5, "")
#rnn.evaluate_model("a")
#neuralNetwork = nn.NeuralNetwork(5, 500, 32, [100,100,100])
#neuralNetwork.train_neural_network(100, "b")
#neuralNetwork.evaluate_model("testModel")

# TESTS TODAY:
# all testing on validation data
# NN: 3 hidden layers, 2 - 512 per layer (9 tests)
# RNN: size 2 - 512 (9 tests)
# 30 epochs (30 min?) *time each test
# each test: accuracy, time

def testNumNodes():
	numEpochs = 40
	# NN tests
	numNodes = 2
	for i in range(9):
		print("NN: numNodes=", numNodes)
		neuralNetwork = nn.NeuralNetwork(5, 500, 32, [numNodes,numNodes,numNodes])
		startTime = datetime.datetime.now()
		neuralNetwork.train_neural_network(numEpochs, "")
		timeToRun = datetime.datetime.now() - startTime
		numNodes *= 2
		print("Time to train: ", timeToRun)
	# RNN tests
	numNodes = 2
	for i in range(9):
		print("RNN: numNodes=", numNodes)
		recurrentNN = rnn.RecurrentNeuralNetwork(5, 50, 10, numNodes, 32)
		startTime = datetime.datetime.now()
		recurrentNN.train_neural_network(numEpochs, "")
		timeToRun = datetime.datetime.now() - startTime
		numNodes *= 2
		print("Time to train: ", timeToRun)
		
testNumNodes()