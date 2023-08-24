import pydub
import tkinter as tk
from tkinter import filedialog

def convert_mp3_to_wav(dst):
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("mp3 files", "*.mp3;"), ("All files", "*.*")])
    if file_path == '':
        print('No file was selected!')
        return None
    
    sound = pydub.AudioSegment.from_mp3(file_path)
    sound.export(dst, format='wav')

if __name__ == '__main__':
    dst = 'test_output/converted_output.wav'
    convert_mp3_to_wav(dst)