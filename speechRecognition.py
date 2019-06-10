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

def parse_args():
	"""
		Parses commandline arguments.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size',
						default=32,
						type=int,
						help='batch size')
	parser.add_argument('--rnn_size',
						default=128,
						type=int,
						help='number of hidden units in RNN')
	parser.add_argument('--output',
						default='',
						type=str,
						help='output directory')
	parser.add_argument('--num_epochs',
						default=10,
						type=int,
						help='number of epochs for training')
	parser.add_argument('--h_layers',
						nargs='+',
						type=int,
						default=[10,10,10],
						help='structure of hidden layers')
	parser.add_argument('--rnn',
						default=False,
						type=bool,
						help='use RNN')
	parser.add_argument('--mlp',
						default=False,
						type=bool,
						help='use MLP')
	parser.add_argument('--train',
						default=False,
						type=bool,
						help='train a model')
	parser.add_argument('--eval',
						default=False,
						type=bool,
						help='evaluate a model')
	parser.add_argument('--input',
						default='',
						type=str,
						help='input directory for the model we are evaluating or .wav file for spectrogram display')
	parser.add_argument('--spectrogram',
						default=False,
						type=bool,
						help='display spectrogram for the features')
	return parser.parse_args()


def main():
	"""
		Takes commands from the command line.
	"""
	args = parse_args()
	if args.train:
		if args.rnn:
			recurrentNN = rnn.RecurrentNeuralNetwork(5, 50, 10, args.rnn_size, args.batch_size)
			rnn.train_neural_network(args.num_epochs, args.output)
		elif args.mlp:
			mlp = nn.NeuralNetwork(5, 500, args.batch_size, args.h_layers)
			neuralNetwork.train_neural_network(args.num_epochs, args.output)
		else:
			print("Please specify type of model you want to train. (See readme.txt)")
	elif args.eval:
		if args.rnn:
			rnn.evaluate_model(args.input)
		elif args.mlp:
			nn.evaluate_model(args.input)
		else:
			print("Please specify type of model you want to evaluate. (See readme.txt)")
	elif args.spectrogram:
		featureExtractor.displaySpectrogram(featureExtractor.computeSpectrogramFromFile(args.input, 50))
	else:
		print("You should choose an option among training, evaluation, and spectrogram display. (See readme.txt)")

if __name__ == '__main__':
    main()
