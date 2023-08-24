import numpy as np
import librosa
import librosa.display


def wav_to_mfcc():

    """
    Convert relevant wave file to MFCC(usable data for neural networks)
    Audio sample is assumed to be 30 seconds long.
    TODO: Data augmentation of a single file to extract more data points.
    TODO: Be able to have user submit wav file. 
    """
    file = input("Enter .wav filepath: ")

    # extract array of amplitude values, 22050 * file time
    signal, sample_rate = librosa.load(file, sr=22050)

    # FFT to power spectrum 
    fft = np.fft.fft(signal)

    #calcuate the overall contribution each frequency has to the overall sound
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sample_rate, len(magnitude))


    #stft to obtain a spectrogram to give us a domain of frequency and time
    stft = librosa.stft(signal)

    spectrogram = np.abs(stft)

    #MFCC last value dictates number
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=2048, hop_length=512, n_mfcc=13)
    return mfcc

wav_file= wav_to_mfcc()
