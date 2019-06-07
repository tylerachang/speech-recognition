"""
	Extracts spectrogram features from .wav files.
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def getFeatures(filename):
	"""
		Computes features for a .wav file, in (n x 1) vector form.
	"""
	
#	# simple features where the desired index is the sum of two features
#	word = filename.split('/')[2]
#	word2idx = {'backward':0,'bed':1, 'bird':2,'cat':3, 'dog':4}
#	index = word2idx[word]
#	return np.array([2*index + 4, -1*index-4])
	
	numWindows = 50
	spectrogram = computeSpectrogramFromFile(filename, numWindows)
	# initialize with zero features per window
	featuresMatrix = np.zeros((numWindows,0))
	# add average amplitude per window
	# featuresMatrix = np.concatenate((featuresMatrix, getAverageAmplitude(spectrogram)), axis=1)
	# add frequencies at 9 percentiles for each window
#	featuresMatrix = np.concatenate((featuresMatrix, getFrequencyPercentiles(spectrogram, 4)), axis=1)

	featuresMatrix = np.concatenate((featuresMatrix, getAverageAmplitudes(spectrogram, 10)), axis=1)

	return featuresMatrix.flatten()

def computeSpectrogramFromFile(filename, numWindows):
	"""
		Computes a spectrogram matrix from a .wav file.
		Output spectrogram has size (numWindows x numFrequencies) where numFrequencies is
		computed within the function based on the length of the audio data and numWindows
		(see inline comments).
	"""
	samplingFrequency, data = wavfile.read(filename)
	windowLength = len(data) // numWindows
	# compute numFrequences because the Fourier transform returns a symmetric vector
	# with length equal to windowLength
	numFrequencies = windowLength // 2
	spectrogram = np.zeros((numWindows, numFrequencies))
	for i in range(numWindows):
		windowData = data[i*windowLength:(i+1)*windowLength]
		transformedData = np.fft.fft(windowData) # compute the discrete Fourier transform using FFT
		frequencyAmplitudes = np.absolute(transformedData) # convert complex numbers to magnitudes
		# because symmetric, take only the first half of the frequency amplitudes
		spectrogram[i] = frequencyAmplitudes[0:numFrequencies]
	return spectrogram

def displayFrequencyAmplitudes(frequencyAmplitudes):
	"""
		Plots frequency amplitudes, where frequencyAmplitudes is a (numFrequencies x 1) vector.
	"""
	plt.plot(frequencyAmplitudes)
	plt.xlabel('Frequency')
	plt.ylabel('Amplitude')
	plt.show()

def displaySpectrogram(spectrogram):
	"""
		Plots a spectrogram using a heatmap, where spectrogram is a (numWindows x numFrequencies)
		matrix.
	"""
	# transpose spectrogram to plot time (window) on x-axis and frequency on y-axis
	spectrogram = spectrogram.T
	heatmap = plt.imshow(spectrogram, cmap = plt.cm.Blues, aspect='auto')
	plt.title("Spectrogram")
	plt.xlabel("Time")
	plt.ylabel("Frequency")
	plt.colorbar(heatmap)
	x_labels = np.arange(0, spectrogram.shape[1], max(spectrogram.shape[1]//10, 1))
	y_labels = np.arange(0, spectrogram.shape[0]//3, max(spectrogram.shape[0]//10, 1))
	plt.xlim(-0.5, spectrogram.shape[1]-0.5)
	plt.ylim(-0.5, spectrogram.shape[0]//3-0.5)
	axes = plt.gca();
	axes.set_xticks(x_labels)
	axes.set_xticklabels(x_labels)
	axes.set_yticks(y_labels)
	axes.set_yticklabels(y_labels)
	plt.show()
	
def getFrequencyPercentiles(spectrogram, numPercentiles):
	"""
		Returns a (numWindows x numPercentiles) matrix containing the frequencies
		at uniformly spaced percentiles.
	"""
	# amplitude between percentiles is the sum along the row divided by numPercentiles+1
	amplitudesBetweenPercentiles = spectrogram.sum(axis=1)/(numPercentiles+1)
	percentileMatrix = np.zeros((spectrogram.shape[0], numPercentiles))
	for window in range(spectrogram.shape[0]):
		currTotal = 0
		currPercentile = 0
		for frequencyIndex in range(spectrogram.shape[1]):
			currTotal += spectrogram[window][frequencyIndex]
			while currTotal > amplitudesBetweenPercentiles[window]:
				# can just insert the frequency index instead of absolute frequency
				percentileMatrix[window][currPercentile] = frequencyIndex
				currTotal -= amplitudesBetweenPercentiles[window]
				currPercentile = min(currPercentile+1, numPercentiles-1)
	return percentileMatrix
	
def getAverageAmplitude(spectrogram):
	"""
		Returns a (numWindows x 1) matrix containing the average amplitude for each window.
	"""
	# sum along rows and divide by the number of columns
	averageAmplitudes = spectrogram.sum(axis=1)/spectrogram.shape[1]
	# reshape to (numWindows x 1) instead of (numWindows, )
	return averageAmplitudes.reshape((spectrogram.shape[0], 1))

def getAverageAmplitudes(spectrogram, numFrequencyBins):
	"""
		Returns a (numWindows x numFrequencyBins) matrix containing the average amplitude
		in each frequency bin for each window.
	"""
	amplitudeMatrix = np.zeros((spectrogram.shape[0], 0))
	numAmplitudesToAverage = spectrogram.shape[1]//numFrequencyBins
	for i in range(numFrequencyBins):
		partialSpectrogram = spectrogram[:,i*numAmplitudesToAverage:(i+1)*numAmplitudesToAverage+1]
		averageAmplitudes = partialSpectrogram.sum(axis=1)/numAmplitudesToAverage
		amplitudeMatrix = np.concatenate((amplitudeMatrix, \
			averageAmplitudes.reshape(averageAmplitudes.shape[0],1)), axis=1)
	return amplitudeMatrix
