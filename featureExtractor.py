"""
	Extracts features from .wav files containing speech.
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft

def readFile(filename, numWindows):
	samplingFrequency, data = wavfile.read(filename)
	windowLength = len(data) // numWindows
	for i in range(numWindows):
		windowData = data[i*windowLength:(i+1)*windowLength]
		transformedData = np.fft.fft(data) # compute the discrete Fourier transform using FFT
		frequencyMagnitudes = np.absolute(transformedData)
		# why symmetric?
		frequencyMagnitudes = frequencyMagnitudes[0:len(frequencyMagnitudes)//2]
	
	plt.plot(frequencyMagnitudes)
	plt.xlabel('frequency')
	plt.ylabel('amplitude')
	plt.show()

readFile("bird.wav", 10)