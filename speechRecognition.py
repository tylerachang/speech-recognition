"""
	Recognizes audio files (.wav format) using neural networks.
"""

import neuralNetworks as nn
import recurrentNeuralNetworks as rnn
import featureExtractor
import datetime
import argparse

def testNumNodes():
	"""
		Tests various numbers of nodes per hidden layer for the multi-layer perceptron.
		Tests various numbers of LSTM units per recurrent layer for the RNN.
	"""
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
	"""
		Trains a sample neural net.
	"""
	# using batch size 32, 3 hidden layers, 128 nodes per hidden layer
	neuralNetwork = nn.NeuralNetwork(5, 500, 32, [128,128,128])
	# 1000 epochs
	neuralNetwork.train_neural_network(1000, "NN")
	
def testRecurrentNeuralNet():
	"""
		Trains a sample recurrent neural net.
	"""
	# using batch size 32, 128 nodes per layer
	recurrentNN = rnn.RecurrentNeuralNetwork(5, 50, 10, 128, 32)
	# 1000 epochs
	recurrentNN.train_neural_network(1000, "RNN")

def main():
	"""
		Takes commands from the command line.
	"""
	
	# let her train NN & RNN, evaluate NN & RNN, display spectrogram
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='an integer for the accumulator')
	parser.add_argument('--sum', dest='accumulate', action='store_const',
					   const=sum, default=max,
					   help='sum the integers (default: find the max)')




	args = parser.parse_args()
	print(args.accumulate(args.integers))
	
	#rnn = rnn.RecurrentNeuralNetwork(5, 50, 10, 128, 100)
#rnn.train_neural_network(5, "")
#rnn.evaluate_model("x")
#neuralNetwork = nn.NeuralNetwork(5, 500, 32, [100,100,100])
#neuralNetwork.train_neural_network(10, "b")
#neuralNetwork.evaluate_model("testModel")
#testNumNodes()
#testNeuralNet()
#testRecurrentNeuralNet()
featureExtractor.displaySpectrogram(featureExtractor.computeSpectrogramFromFile("dog0.wav", 50))