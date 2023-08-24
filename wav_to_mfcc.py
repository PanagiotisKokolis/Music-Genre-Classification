import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

FILE = "country.00002.wav"

# extract array of amplitude values, 22050 * file time
signal, sample_rate = librosa.load(FILE, sr=22050)

# visualize the above array
plt.figure(figsize= (12,5))
librosa.display.waveshow(signal)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("waveform")

# FFT to power spectrum swap
fft = np.fft.fft(signal)

#calcuate the overall contribution each frequency has to the overall sound
magnitude = np.abs(fft)
frequency = np.linspace(0, 22050, len(magnitude))

folded_mangnitude = magnitude[:int(len(magnitude)/2)]
folded_frequency = frequency[:int(len(frequency)/2)]

#visualize power spectrum array
plt.plot(folded_frequency,folded_mangnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power Spectrum")

#stft to obtain a spectrogram to give us a domain of frequency and time
stft = librosa.stft(signal)

spectrogram = np.abs(stft)

#apply log to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

#visualize spectrogram
plt.figure(figsize= (12,5))
librosa.display.specshow(log_spectrogram)
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency")
plt.colorbar(format="%+23.0f dB")
plt.title("Spectrogram")

#MFCC last value dictates number
MFCC = librosa.feature.mfcc(y=signal, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13)

#visualize MFCCs
plt.figure(figsize= (12,5))
librosa.display.specshow(MFCC)
plt.xlabel("Time")
plt.ylabel("MFCC coefficents")
plt.colorbar()
plt.title("MFCC")

plt.show()