"""
	Implementation of a neural network using TensorFlow (v1).
	Implements a basic feedforward NN.
	
	Code tutorial:
	https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/
"""

import tensorflow as tf
import batches
import numpy as np
import datetime

class NeuralNetwork:
	
	def __init__(self, n_classes, n_features, batch_size, n_hidden_nodes = [10,10,10]):
		print("Initializing neural net.")
		tf.reset_default_graph()
		self.n_hidden_nodes = n_hidden_nodes
		self.n_hidden_layers = len(self.n_hidden_nodes)
		
		self.n_classes = n_classes
		self.n_examples = batches.getTotalNumFiles() # get total number of training files
		# batch size should divide n_examples to ensure all data is included, but should
		# not throw any errors
		self.batch_size = batch_size
		self.n_features = n_features
		
		print("Loading features.")
		self.trainingData = batches.TrainingData(self.n_features)
		print("Loaded features.")

	def neural_network_model(self, data):
		"""
			Defines the neural network model.
		"""
		# initialize weights and biases as zero
		hidden_layers = []
		hidden_layers.append( \
			{'weights':tf.Variable(tf.random_normal([self.n_features, self.n_hidden_nodes[0]])), \
			 'biases':tf.Variable(tf.random_normal([self.n_hidden_nodes[0]]))})
		for i in range(self.n_hidden_layers-1):
			hidden_layers.append( \
				{'weights':tf.Variable(tf.random_normal( \
					[self.n_hidden_nodes[i], self.n_hidden_nodes[i+1]])), \
				'biases':tf.Variable(tf.random_normal([self.n_hidden_nodes[i+1]]))})
		output_layer = \
			{'weights':tf.Variable(tf.random_normal([self.n_hidden_nodes[-1], self.n_classes])), \
			 'biases':tf.Variable(tf.random_normal([self.n_classes]))}
		
		for i in range(self.n_hidden_layers):
			data = tf.add(tf.matmul(data,hidden_layers[i]['weights']), hidden_layers[i]['biases'])
			# use sigmoid instead of relu because relu results in very large weights
			data = tf.nn.sigmoid(data)
		output = tf.matmul(data,output_layer['weights']) + output_layer['biases']
		return output

	def train_neural_network(self, n_epochs, output_dir = ""):
		"""
			Trains the neural network model.
		"""
		x = tf.placeholder('float', [None, self.n_features])
		y = tf.placeholder('float', [None, self.n_classes])
		prediction = self.neural_network_model(x)
		# use softmax cross entropy as loss
		cost = tf.reduce_mean( \
			tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
		# use Adam optimizer instead of stochastic gradient descent
		# default learning rate is 0.001
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for epoch in range(n_epochs):
				epoch_loss = 0
				n_batches = self.n_examples//self.batch_size
				epoch_x_list, epoch_y_list = \
					self.trainingData.getTrainingBatches(self.batch_size, n_batches)
				startTime = datetime.datetime.now()
				for i in range(n_batches):
					epoch_x = epoch_x_list[i]
					epoch_y = epoch_y_list[i]
					_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
					epoch_loss += c
				epoch_loss /= n_batches
				epochTime = datetime.datetime.now() - startTime
				print("Epoch time: ", epochTime)
				print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

				# print the accuracy on the validation data
				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				val_x, val_y = batches.getValidationData() # change this back
				print('Validation accuracy:', accuracy.eval({x: val_x, y: val_y}))

			# save the model
			if output_dir != "":
				tf.saved_model.simple_save(sess, output_dir, \
					inputs={'x':x}, outputs={'prediction':prediction})
				print("Model saved to: ", output_dir)

def evaluate_model(model_dir):
	"""
		Evaluates the accuracy of a saved model on the test data.
	"""
	predictFunc = tf.contrib.predictor.from_saved_model(model_dir)
	test_x, test_y = batches.getTestData()
	prediction = predictFunc({'x':test_x})['prediction']
	numCorrect = 0
	for i in range(test_x.shape[0]):
		if np.argmax(prediction[i]) == np.argmax(test_y[i]):
			numCorrect += 1
	print('Test accuracy:', float(numCorrect)/test_x.shape[0])

