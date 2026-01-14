import scipy.io.wavfile as wavfile
from scipy.fft import fft, fftfreq
import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, argrelmax
from scipy.signal import savgol_filter
import argparse
import csv
import pretty_errors
import pymeshlab
import pandas as pd

#adoped from https://github.com/Metallicode/RandomProjects_IOT/blob/master/06_fft_analysis/python_fft.py
#added loop to covert into xyz file 
#run like python fft.py > out.xyz then open in mesh lab and run commands listed in writeup 


def normalizeRange(x, multiply=1.0, add=0.0):
	minX = x.min()
	maxX = x.max()

	return np.add(np.multiply(np.divide(np.subtract(x, minX), np.subtract(maxX, minX)), multiply), add)


def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return x, y


def pol2Cart3d(r, polar, alpha):
	x = r * np.sin(polar) * np.cos(alpha)
	y = r * np.sin(polar) * np.sin(alpha)
	z = r * np.cos(polar)

	return x, y, z


def rotatePoints(x, y, z):

	minX = min(x)
	maxX = max(x)
	scale = maxX - minX

	x = np.multiply(np.divide(np.subtract(x, minX), scale), 2 * np.pi)

	x, z = pol2cart(z, x)

	x = np.reshape(x, (-1, 1))
	y = np.reshape(y, (-1, 1))
	z = np.reshape(z, (-1, 1))

	return np.concatenate([x, y, z], axis=1)


def fftSignal(signal, s_rate):
	FFT = fft(signal)
	freqs = fftfreq(len(FFT), (1.0 / s_rate))

	FFT = 2.0 / len(FFT) * np.abs(FFT[1:len(FFT) // 2])
	freqs = freqs[1:len(freqs) // 2]

	return FFT, freqs


def toroidMusic(signal, s_rate, segments):

	songSections = np.array_split(signal, segments)
	print(len(songSections[0]))
	r = []
	polar = []
	alpha = []

	for idx, section in enumerate(songSections):

		FFT, freqs = fftSignal(section, s_rate)
		#FFT = np.log10(FFT)
		fftIndicies = argrelmax(FFT, order=1000)
		#fftIndicies = find_peaks(FFT, prominence=1)
		#print(fftIndicies)

		if len(fftIndicies[0]) > 1:

			#plt.plot(freqs, FFT)
			#plt.plot(freqs[fftIndicies], FFT[fftIndicies])
			#plt.show()

			FFT = FFT[fftIndicies]
			freqs = freqs[fftIndicies]

			print("%s : %s - %s" % (idx, min(FFT), max(FFT)))

			polarSection = normalizeRange(freqs, np.pi)  # Normalize frequencies from 0 to pi
			alphaSection = [2.0 * np.pi * (idx / segments)] * len(FFT)

			#fig = plt.figure()
			#ax = fig.add_subplot(projection="3d")
			#x, y, z = pol2Cart3d(FFT, polarSection, alphaSection)
			#ax.scatter(x, y, z)
			#plt.show()


			polar.extend(polarSection)
			alpha.extend(alphaSection)
			r.extend(FFT)

	#plt.show()
	x, y, z = pol2Cart3d(r, polar, alpha)

	#ax.scatter(x, y, z)
	#plt.show()

	x = np.reshape(x, (-1, 1))
	y = np.reshape(y, (-1, 1))
	z = np.reshape(z, (-1, 1))

	return np.concatenate([x, y, z], axis=1)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--wav", type=str, help="path to WAV file")

args = parser.parse_args()
path = args.wav
file = path.split("\\")[-1].split(".")[0]

print("Computing FFT")
s_rate, signal = wavfile.read(args.wav)

if signal.shape[1] == 2:
	print("\tConverting stereo WAV to mono")
	signal = signal.sum(axis=1) / 2.0  # Mix down to mono

xyzData = toroidMusic(signal, s_rate, 22)

print("Writing data to file")
with open("%s.xyz" % file, "w", newline="") as xyzFile:
	writer = csv.writer(xyzFile, delimiter=" ")
	writer.writerows(xyzData)

print("plotting in pymesh")
ms = pymeshlab.MeshSet()
ms.load_new_mesh("%s.xyz" % file)

print("normalizing points")
ms.compute_normal_for_point_clouds()

print("poisson points")
ms.generate_surface_reconstruction_screened_poisson()

print("Save STL")
ms.save_current_mesh("%s_script.obj" % file)
