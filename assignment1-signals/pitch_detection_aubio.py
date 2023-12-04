import aubio
import matplotlib.pyplot as plt
import os
from scipy.fft import fft
import scipy
import numpy as np

# Constants for the audio stream
BUFFER_SIZE = 4 * 1024
CHANNELS = 1
# RATE = 44100
RATE = 48000

# Create an Aubio pitch detection object
pDetection = aubio.pitch("default", BUFFER_SIZE, BUFFER_SIZE, RATE)
pDetection.set_unit("Hz")
pDetection.set_tolerance(0.9)

# Create a figure and an axis for the plot
# fig, ax = plt.subplots()
# x = list(range(100))  # Adjust as needed
# y = [0]*100  # Adjust as needed
# line, = ax.plot(x, y)

# Initialize a list to store the pitch values
pitches = []

# Read from an audio file
# path = "/home/luka/UPEC/courses/software_integration/assignment1-signals/audio_inputs/octave"
path = "/home/luka/UPEC/courses/software_integration/assignment1-signals/audio_inputs/whatsapp"
print(path)
# save all wav files in a directory
files = []
for file in os.listdir(path):
    if file.endswith(".wav"):
        files.append(file)
audio_idx = 5

# filename = "piano-a_A_major.wav"
filename = files[audio_idx]
# filename = "/home/luka/UPEC/courses/software_integration/assignment1-signals/audio_inputs/online_piano.wav"
source = aubio.source(os.path.join(path, filename), RATE, BUFFER_SIZE)

# Process each audio buffer from the file
while True:
    samples, read = source()
    pitch = pDetection(samples)[0]
    print(f"Pitch: {pitch:.2f} Hz. Note: {aubio.freq2note(pitch):s}")

    if read < source.hop_size:
        break

# plt.show()
print(filename)
# print(samples.shape)
# print(samples.dtype)
# print(f'Min: {samples.min():.2f}, Max: {samples.max():.2f}')

# plot the fft of the audio signal
# Number of sample points
# N = 44100
# # sample spacing
# T = 1.0 / 44100.0
# x = np.linspace(0.0, N * T, N, endpoint=False)
# sr, audio = scipy.io.wavfile.read(os.path.join(path, filename))
# print(f"Number of channels: {audio.shape[1]}")
# print(sr)
# audio = audio[:, 0]
# yf = fft(audio)
# xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)


# fig, ax = plt.subplots()
# magnitude = 2.0 / N * np.abs(yf[: N // 2])
# ax.plot(xf, magnitude)
# plt.show()
