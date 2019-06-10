CS321 AI Final Project: Speech Recognition
Authors: Rosa Zhou and Tyler Chang

To run our code, you can run speechRecognition.py from the command line (see below).

Our dataset can be downloaded from: https://www.tensorflow.org/tutorials/sequences/audio_recognition (click the link for the Speech Commands dataset). The data (containing subdirectories for each word) should be placed in a folder called "data" in the same directory as speechRecognition.py.

Command line arguments:
--spectrogram	A boolean for whether to display a spectrogram.
--rnn			A boolean for whether to use an RNN.
--mlp			A boolean for whether to use an MLP.
--train			A boolean for whether to train the model.
--eval			A boolean for whether to evaluate a saved model on the test set (you still have
				to specify whether the saved model is an RNN or MLP).
Note: you can only select one of spectrogram, train, and eval; you can only select one of rnn and mlp.

Options:
--batch_size	The batch size for training.
--rnn_size		The number of hidden units in each LSTM cell for an RNN.
--output		The output directory.
--input			The input directory containing a saved model or a .wav
				file path for displaying a spectrogram.
--num_epochs	The number of epochs to train.
--h_layers		A sequence of integers (each separated by a space) for
				the number of hidden nodes in each layer.


The trained MLP and RNN described in our paper (1000 epochs, 128 nodes per layer, batch size 32) can be found in the directories "NN" and "RNN" respectively.
If you want to modify the feature extraction for the audio, you can modify the getFeatures() function in featureExtractor.py (then remember to change the number of features when initializing the NNs in speechRecognition.py).