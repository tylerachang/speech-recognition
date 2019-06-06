"""
	Extracts spectrogram features from .wav files.
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

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
	displaySpectrogram(spectrogram)
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
	y_labels = np.arange(0, spectrogram.shape[0], max(spectrogram.shape[0]//10, 1))
	plt.xlim(-0.5, spectrogram.shape[1]-0.5)
	plt.ylim(-0.5, spectrogram.shape[0]-0.5)
	axes = plt.gca();
	axes.set_xticks(x_labels)
	axes.set_xticklabels(x_labels)
	axes.set_yticks(y_labels)
	axes.set_yticklabels(y_labels)
	plt.show()
	
# TODO: extract features from spectrogram
	
computeSpectrogramFromFile("bird0.wav", 50)
computeSpectrogramFromFile("bird1.wav", 50)
computeSpectrogramFromFile("bird2.wav", 50)
computeSpectrogramFromFile("cat0.wav", 50)
computeSpectrogramFromFile("cat1.wav", 50)
computeSpectrogramFromFile("cat2.wav", 50)
computeSpectrogramFromFile("backward0.wav", 50)
computeSpectrogramFromFile("backward1.wav", 50)
computeSpectrogramFromFile("backward2.wav", 50)