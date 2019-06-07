"""
	Implementation of an RNN using tensorflow.
"""

import tensorflow as tf
import speechRecognition
import numpy as np

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 5
batch_size = 100
n_features = 50
n_examples = 1000

x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float')

def neural_network_model(data):
	"""
		TODO: ASK ANNA ABOUT THIS FUNCTION!!!! We just copied and pasted???
		Maybe generalize to using more hidden layers.
		Clean this code!!
	"""
	# initialize weights and biases as zero
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	
	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	#TODO: ask Anna why use reduce_mean and what this line means
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
	# use Adam optimizer instead of stochastic gradient descent
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
    
	n_epochs = 30
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for i in range(int(n_examples/batch_size)):
				epoch_x, epoch_y = speechRecognition.getBatch(batch_size)
				j, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		test_x, test_y = speechRecognition.getTestData(batch_size)
		print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)