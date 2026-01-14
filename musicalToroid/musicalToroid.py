import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, argrelmax
from scipy.signal import savgol_filter
import argparse
from generateRings import *
import pretty_errors

#TODO: if no peaks are found, fill with set value

def fftSignal(signal, s_rate):
	FFT = fft(signal)
	freqs = fftfreq(len(FFT), (1.0 / s_rate))

	FFT = 2.0 / len(FFT) * np.abs(FFT[1:len(FFT) // 2])
	freqs = freqs[1:len(freqs) // 2]

	return FFT, freqs


def toroidMusic(signal, s_rate, segments, ringRad):

	keepPeaks = 0
	verticiesOut = []
	facesOut = []

	songSections = np.array_split(signal, segments)
	print("Number of points in song section: %s" % len(songSections[0]))

	for idx, section in enumerate(songSections):

		FFT, freqs = fftSignal(section, s_rate)

		keepI = np.argwhere(freqs < 10000)  # Keep indexes where frequency value is less than 10k
		FFT = FFT[keepI]
		freqs = freqs[keepI]

		fftIndicies = argrelmax(FFT, order=50)  # Find local peaks

		if keepPeaks == 0:
			keepPeaks = int(0.9 * len(fftIndicies[0]))  # Set the peak count

		fftIndicies = (fftIndicies[0][:keepPeaks],)  # Set indexes of peaks to keep

		if len(fftIndicies[0]) > 1:

			print("%s : %s - %s : %s : %s" % (idx, min(FFT), max(FFT), len(FFT), len(fftIndicies[0])))
			plt.plot(freqs, FFT)
			plt.plot(freqs[fftIndicies], FFT[fftIndicies])
			plt.show()

			#FFT = FFT[fftIndicies]
			#freqs = freqs[fftIndicies]

			rotation = [2.0 * np.pi * (idx / segments)]  # calculate the rotation radian about z-axis
			verticiesOut.extend(genVertices(freqs[fftIndicies], FFT[fftIndicies], [0, 10000], rotation, ringRad=ringRad))
			facesOut.extend(genFaces(idx, segments, keepPeaks))

	return np.array(verticiesOut), np.array(facesOut)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--wav", type=str, help="path to WAV file")
parser.add_argument("-s", "--seg", type=int, default=20, help="Number of segments about Z-Axis")
parser.add_argument("-r", "--rng", type=int, default=5,  help="Radius of internal toroid")

args = parser.parse_args()
path = args.wav
segments = args.seg
ringRad = args.rng
file = path.split("\\")[-1].split(".")[0]

print("Computing FFT")
s_rate, signal = wavfile.read(args.wav)

if signal.shape[1] == 2:
	print("\tConverting stereo WAV to mono")
	signal = signal.sum(axis=1) / 2.0  # Mix down to mono

verticies, faces = toroidMusic(signal, s_rate, segments, ringRad)
genSTL(verticies, faces, file)
