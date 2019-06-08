"""
	Recognizes audio files (.wav format) using neural networks.
"""

import neuralNetworks as nn

neuralNetwork = nn.NeuralNetwork(5, 500, 100, [100,100,100])
neuralNetwork.train_neural_network(1, "testModel")
neuralNetwork.evaluate_model("testModel")