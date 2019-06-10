"""
	Implementation of an LSTM neural network using TensorFlow (v1).
	Uses out-of-the-box LSTM cell implementations.
	
	Code tutorial:
	https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/
"""

import tensorflow as tf
import batches
import numpy as np
import datetime

class RecurrentNeuralNetwork:
	
	def __init__(self, n_classes, n_chunks, chunk_size, rnn_size, batch_size):
		print("Initializing RNN.")
		tf.reset_default_graph()
		self.rnn_size = rnn_size
			
		self.n_classes = n_classes
		self.n_examples = batches.getTotalNumFiles() # get total number of training files
		# batch size should divide n_examples to ensure all data is included, but should
		# not throw any errors
		self.batch_size = batch_size
		self.n_chunks = n_chunks
		self.chunk_size = chunk_size
		
		print("Loading features.")
		self.trainingData = batches.TrainingData(self.n_chunks * self.chunk_size)
		print("Loaded features.")

	def recurrent_neural_network_model(self, data):
		"""
			Defines the recurrent neural network model.
		"""
		outputLayer = {'weights':tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
				 'biases':tf.Variable(tf.random_normal([self.n_classes]))}

		# transpose such that dim 0 is time window, dim 1 is example index, and
		# dim 2 is feature (for the example at the given time)
		data = tf.transpose(data, perm=[1,0,2])
		# flatten so that each row is a chunk
		data = tf.reshape(data, [-1, self.chunk_size])
		# create a list of (n_chunks x chunk_size) matrices (each matrix
		# corresponds with one example)
		data = tf.split(data, self.n_chunks, 0)

		lstm_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, state_is_tuple=True)
		
		# We usually need to construct a computation graph first in TensorFlow.
		# Static rnn builds a static computational graph with a fixed number of time steps, but
		# dynamic rnn would build the computational graph as it executed and thus would
		# allow input sequences with variable length.
		outputs, states = tf.nn.static_rnn(lstm_cell, data, dtype=tf.float32)

		output = tf.matmul(outputs[-1],outputLayer['weights']) + outputLayer['biases']
		return output

	def train_neural_network(self, n_epochs, output_dir = ""):
		"""
			Trains the recurrent neural network model.
		"""
		x = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
		y = tf.placeholder('float', [None, self.n_classes])
		prediction = self.recurrent_neural_network_model(x)
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
					# reshape epoch_x to represent each example in chunks
					epoch_x = epoch_x.reshape( \
						(self.batch_size, self.n_chunks, self.chunk_size))
					_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
					epoch_loss += c
				epoch_loss /= n_batches
				epochTime = datetime.datetime.now() - startTime
				print("Epoch time: ", epochTime)
				print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

				# print the accuracy on the validation data
				correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
				val_x, val_y = batches.getValidationData()
				val_x = val_x.reshape((-1, self.n_chunks, self.chunk_size))
				print('Validation accuracy:', accuracy.eval({x: val_x, y: val_y}))
			
			# save the model
			if output_dir != "":
				tf.saved_model.simple_save(sess, output_dir, \
					inputs={'x':x}, outputs={'prediction':prediction})
				print("Model saved to: ", output_dir)

	def evaluate_model(self, model_dir):
		"""
			Evaluates the accuracy of a saved model on the test data.
		"""
		predictFunc = tf.contrib.predictor.from_saved_model(model_dir)
		test_x, test_y = batches.getTestData()
		test_x = test_x.reshape((-1, self.n_chunks, self.chunk_size))
		prediction = predictFunc({'x':test_x})['prediction']
		numCorrect = 0
		for i in range(test_x.shape[0]):
			if np.argmax(prediction[i]) == np.argmax(test_y[i]):
				numCorrect += 1
		print('Test accuracy:', float(numCorrect)/test_x.shape[0])

