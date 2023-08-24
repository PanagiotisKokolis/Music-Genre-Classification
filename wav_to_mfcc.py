import numpy as np
import librosa
import librosa.display
import tkinter as tk
from tkinter import filedialog

def wav_to_mfcc():

    """
    Convert relevant wave file to MFCC(usable data for neural networks)
    Audio sample is assumed to be 30 seconds long.
    TODO: Data augmentation of a single file to extract more data points.
    TODO: Be able to have user submit wav file. 
    """
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3;*.wav"), ("All files", "*.*")])
    if file_path == '':
        print('No file was selected!')
        return None

    signal, sample_rate = librosa.load(file_path, sr=22050)
    
    samples_per_track = 22050 * 30
    samples_per_segment = samples_per_track // 10
    mfccs = []
    
    for i in range(10):
        start = i * samples_per_segment
        finish = start + samples_per_segment

        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_fft=2048, hop_length=512, n_mfcc=13)
        mfcc = mfcc.T
        mfccs.append(mfcc)

    return mfccs

if __name__ == '__main__':
    mfcc = wav_to_mfcc()